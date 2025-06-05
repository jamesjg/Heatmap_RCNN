import numpy as np
from lib.dataset.head import decode_head
from lib.model import *
from lib.utils.loss import AverageMeter,cal_batch_ION,compute_fr_and_auc, interval_values
import time
import torch
from torch.nn import functional as F

def get_model(config):
    if config.model.backbone in ["hourglass"]:
        if hasattr(config.model, "use_roi") and config.model.use_roi>1:
            return hgnet_multi_out(config)
        else:
            return hgnet(config)
    elif config.model.backbone in ["resnet18", "resnet34", "resnet50"]:
        return ResNet_FC(config)
    else:
        raise ValueError("No suitable model now")

def get_roi(config, feature, pred_landmarks, stage=1):
    """All tensors are on the GPU device"""
    # 32,98,64,64 -> 32, 98, roi_size, roi_size

    batch = feature.size(0)
    num_landmarks = feature.size(1)
    roi_size = config.model.roi_size if hasattr(config.model, "roi_size") else config.model.roi_sizes[config.model.output_stages.index(stage)]
    half_roi_size = (roi_size - 1) // 2

    pad_feature = torch.nn.ZeroPad2d(half_roi_size)(feature)
    pad_pred_landmarks = pred_landmarks + torch.tensor([half_roi_size,half_roi_size]).cuda(config.train.gpu_id) # the new landmark location (roi center) after padding
    pad_lt_landmarks = pad_pred_landmarks - half_roi_size # the left-top location of the proposal
    pad_rb_landmarks = pad_pred_landmarks + half_roi_size # the right-bottom location of the proposal
    pad_rb_landmarks += 1 # indices setting

    # heatmap features 32,98,64,64 -> 32,98,7,7
    x_coor = torch.arange(0,pad_feature.size(-1)).cuda(config.train.gpu_id).expand(batch,num_landmarks,pad_feature.size(-1))
    y_coor = torch.arange(0,pad_feature.size(-2)).cuda(config.train.gpu_id).expand_as(x_coor)
    x_mask = torch.mul(x_coor>=pad_lt_landmarks[:,:,0:1], x_coor<pad_rb_landmarks[:,:,0:1]).unsqueeze(-2).expand(batch,num_landmarks,pad_feature.size(-2),pad_feature.size(-1)) # rb has added 1
    y_mask = torch.mul(y_coor>=pad_lt_landmarks[:,:,1:2], y_coor<pad_rb_landmarks[:,:,1:2]).unsqueeze(-1).expand_as(x_mask) # rb has added 1
    roi_mask = torch.mul(x_mask,y_mask)
    roi_features = pad_feature[roi_mask].view(batch,num_landmarks,roi_size,roi_size).detach()

    return roi_features


def get_use_roi_and_resolution_weight(config):

    if hasattr(config.model, "use_roi"):
        use_roi = config.model.use_roi
    else:
        use_roi = 0

    if hasattr(config.model, "output_stages"):
        assert len(config.model.output_stages) == use_roi, "use_roi is not match for the length of output_stages"
        assert len(config.model.multi_stage_sigmas) == use_roi, "use_roi is not match for the length of multi_stage_sigmas"
        assert len(config.model.sup_losses) == use_roi, "use_roi is not match for the length of sup_losses"
        assert len(config.model.stage_heatmap_weights) == use_roi, "use_roi is not match for the length of stage_heatmap_weights"
        assert len(config.model.roi_sizes) == use_roi, "use_roi is not match for the length of roi_sizes"
    
    if use_roi > 1:
        resolution_weights = config.model.stage_heatmap_weights
        resolution_weights[0] = 1 / (config.backbone.num_stack - 1) if config.backbone.num_stack > 1 else 1.0
    elif use_roi == 1:
        resolution_weights = [1 / (config.backbone.num_stack - 1)] if config.backbone.num_stack > 1 else 1.0
    else:
        resolution_weights = None
    
    return use_roi, resolution_weights


def train_model(train_info, logger):
    config = train_info["config"]
    model = train_info["model"]
    optimizer = train_info["optimizer"]
    criterion = train_info["criterion"]
    train_loader = train_info["dataloader"]
    device = train_info["device"]
    is_train_roi = not train_info["train_backbone"]
    cal_loss_fn = get_cal_loss_fn(config)
    use_roi, resolution_weights = get_use_roi_and_resolution_weight(config)
    start_time = time.time()

    losses = AverageMeter()
    losses.reset()
    freq_losses = AverageMeter()
    freq_losses.reset()

    model.train()

    for i,(imgs,target_maps,landmarks,multi_target_maps) in enumerate(train_loader):
        imgs = imgs.to(device)
        target_maps = target_maps.to(device)
        multi_target_maps = [stage_target_maps.to(device) for stage_target_maps in multi_target_maps]

        output = model(imgs)
        loss = cal_loss_fn(output,target_maps,multi_target_maps,criterion,model.training,resolution_weights)

        if is_train_roi and use_roi == 1:
            landmarks = landmarks.to(device)
            last_stage_output = output[:,-1] # 在训练时输出stack的heatmaps， 取最后一组heatmap
            pred_landmarks = decode_head(last_stage_output)
            roi_feature = get_roi(config, last_stage_output, pred_landmarks)
            roi_offset = model.forward_roi(roi_feature)
            loss += criterion["roi_loss"](roi_offset, landmarks-pred_landmarks) # here need no weight
        elif is_train_roi and use_roi > 1:
            landmarks = landmarks.to(device)
            max_resolution_pred_landmarks, concate_roi_features, resolution_landmark_offsets = get_multi_rois(config, output[1])
            roi_offset = model.forward_roi(concate_roi_features, resolution_landmark_offsets = resolution_landmark_offsets)
            loss += criterion["roi_loss"](roi_offset, landmarks-max_resolution_pred_landmarks) # here need no weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))
        freq_losses.update(loss.item(), imgs.size(0))

        if config.train.freq is not None and i % config.train.freq == 0:
            logger.info(f"Train Info: [ {i} ] iters AVG Loss: {freq_losses.avg}")
            freq_losses.reset()

    logger.info(f"Train Info: All AVG Loss: {losses.avg:.6f}] | Time cost: {(time.time()-start_time):.2f}s")

    return losses.avg


def finetune_roi(train_info, logger):
    config = train_info["config"]
    model = train_info["model"]
    optimizer = train_info["optimizer"]
    criterion = train_info["criterion"]
    train_loader = train_info["dataloader"]
    device = train_info["device"]
    cal_loss_fn = get_cal_loss_fn(config)
    use_roi, resolution_weights = get_use_roi_and_resolution_weight(config)
    start_time = time.time()

    losses = AverageMeter()
    losses.reset()
    freq_losses = AverageMeter()
    freq_losses.reset()

    model.eval()
    model.roi_module.train()

    for i,(imgs,target_maps,landmarks,multi_target_maps) in enumerate(train_loader):
        imgs = imgs.to(device)
        target_maps = target_maps.to(device)
        multi_target_maps = [stage_target_maps.to(device) for stage_target_maps in multi_target_maps]

        with torch.no_grad():
            output = model(imgs)
            loss = 0 # cal_loss_fn(output,target_maps,multi_target_maps,criterion,model.training,resolution_weights)

        if use_roi == 1:
            landmarks = landmarks.to(device)
            pred_landmarks = decode_head(output)
            roi_feature = get_roi(config, output, pred_landmarks)
            roi_offset = model.forward_roi(roi_feature)
            loss += criterion["roi_loss"](roi_offset, landmarks-pred_landmarks) # here need no weight
        elif use_roi > 1:
            landmarks = landmarks.to(device)
            max_resolution_pred_landmarks, concate_roi_features, resolution_landmark_offsets = get_multi_rois(config, output)
            roi_offset = model.forward_roi(concate_roi_features, resolution_landmark_offsets=resolution_landmark_offsets)
            loss += criterion["roi_loss"](roi_offset, landmarks-max_resolution_pred_landmarks) # here need no weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))
        freq_losses.update(loss.item(), imgs.size(0))

        if config.train.freq is not None and i % config.train.freq == 0:
            logger.info(f"Train Info: [ {i} ] iters AVG Loss: {freq_losses.avg}")
            freq_losses.reset()

    logger.info(f"Train Info: All AVG Loss: {losses.avg:.6f}] | Time cost: {(time.time()-start_time):.2f}s")

    return losses.avg


def get_multi_rois(config, multi_stage_heatmaps):
    multi_stage_pred_landmarks, multi_stage_roi_features = [], []
    output_stages = config.model.output_stages
    multi_resolutions = [int(64 / 2 ** (r-1)) for r in output_stages]
    low_decode_type = config.model.low_decode_type if hasattr(config.model, "low_decode_type") else 0

    if low_decode_type == 0:
        # 以最大分辨率的预测坐标值 降采样到不同分辨率上
        max_first_stage_pred_landmark = decode_head(multi_stage_heatmaps[0])
        max_resolution_multi_stage_pred_landmarks = [(max_first_stage_pred_landmark / 
                                                    2 ** (stage - output_stages[0])).int() 
                                                    for stage in output_stages]  

        multi_maxres_stage_roi_features = [] 
        for stage_heatmap, stage_maxres_pred_landmarks, stage in zip(multi_stage_heatmaps, max_resolution_multi_stage_pred_landmarks, output_stages):
            stage_maxres_roi_features = get_roi(config, stage_heatmap, stage_maxres_pred_landmarks,stage)
            stage_maxres_roi_features_flat = stage_maxres_roi_features.view(stage_maxres_roi_features.size(0),stage_maxres_roi_features.size(1), -1)
            multi_maxres_stage_roi_features.append(stage_maxres_roi_features_flat)

        multi_maxres_stage_roi_features = torch.concat(multi_maxres_stage_roi_features, -1).detach()

        return max_first_stage_pred_landmark, multi_maxres_stage_roi_features, 0

    elif low_decode_type == 1:
        # 解码所有分辨率,采用各自的landmarks提取ROI
        multi_private_stage_roi_features = []
        for stage_heatmap,stage in zip(multi_stage_heatmaps, output_stages):
            stage_pred_landmarks = decode_head(stage_heatmap)
            multi_stage_pred_landmarks.append(stage_pred_landmarks)
            stage_private_roi_features = get_roi(config, stage_heatmap, stage_pred_landmarks, stage)
            stage_private_roi_features_flat = stage_private_roi_features.view(stage_private_roi_features.size(0),stage_private_roi_features.size(1), -1)
            multi_private_stage_roi_features.append(stage_private_roi_features_flat)

        multi_private_stage_roi_features = torch.concat(multi_private_stage_roi_features, -1)
        multi_private_stage_pred_landmarks_offset = [multi_stage_pred_landmarks[0] - stage_pred_landmarks*(2**(stage-1))  for stage, stage_pred_landmarks in zip(output_stages[1:], multi_stage_pred_landmarks[1:])]

        return multi_stage_pred_landmarks[0], multi_private_stage_roi_features, multi_private_stage_pred_landmarks_offset

    """
    # TODO 采用平均的landmarks
    # calculate average 
    average_landmarks_list = [stage_pred_landmarks/resolution_size 
                        for stage_pred_landmarks, resolution_size in 
                        zip(multi_stage_pred_landmarks, multi_resolutions)]
    average_landmarks_np = np.array(average_landmarks_list).mean(axis=0)
    multi_average_landmarks = [ (average_landmarks_np*resolution_size).int() for resolution_size in multi_resolutions]
    for stage_heatmap,stage in zip(multi_stage_heatmaps, output_stages):
        pass    
    """






def get_cal_loss_fn(config):

    if config.model.backbone in ["hourglass"]:
        if hasattr(config.model, "use_roi") and config.model.use_roi > 1:
            return hourglass_roi_process
        else:
            return hourglass_process

    elif config.model.backbone in ["resnet18", "resnet34", "resnet50"]:
        return resnet_fc_process

    else:
        raise ValueError("No suitable function now")


def hourglass_process(output, targets, multi_stage_targets, criterion, training=False, weights=None):
    if len(targets.shape) != len(output.shape):
        target_expand_maps = targets.unsqueeze(1).expand_as(output)
    else:
        target_expand_maps = targets
    loss = criterion["loss"](output, target_expand_maps)
    return loss

def hourglass_roi_process(outputs, targets, multi_stage_targets, criterion, training=False, weights=None):
    # 既有 stack hourglass监督（此处stack-1）， 又有multi stage heatmap监督（最后stack输出多个分辨率）
    loss = 0
    if training:
        stack_heatmaps, multi_stage_heatmaps = outputs 
        if len(stack_heatmaps): # when stack = 1, the output is empty
            loss += hourglass_process(stack_heatmaps, targets, multi_stage_targets, criterion, training, weights)
    else:
        multi_stage_heatmaps = outputs

    loss += resnet_fc_process(multi_stage_heatmaps, targets, multi_stage_targets, criterion, training, weights)

    return loss



def resnet_fc_process(multi_stage_outputs, targets, multi_stage_targets, multi_criterion, training=False, weights=None):
    # outputs: stages outout and FPN pool 
    loss = 0
    stage_criterions = multi_criterion["sup_losses"]
    for stage_i in range(len(multi_stage_targets)):
        criterion = stage_criterions[stage_i]
        stage_target = multi_stage_targets[stage_i]
        stage_pred = multi_stage_outputs[stage_i]
        loss += weights[stage_i] * criterion(stage_pred, stage_target)
    
    return loss

def val_model(val_info, logger):
        config = val_info["config"]
        model = val_info["model"]
        criterion = val_info["criterion"]
        val_loader = val_info["dataloader"]
        device = val_info["device"]
        is_val_roi = not val_info["val_backbone"]
        cal_loss_fn = get_cal_loss_fn(config)
        decode_heatmap_fn = get_decode_fn(config)
        num_landmarks = config.data.num_landmarks
        gt_heatmap_size = config.heatmap.heatmap_size
        use_roi, resolution_weights = get_use_roi_and_resolution_weight(config)
        start_time = time.time()

        losses = AverageMeter()
        losses.reset()

        model.eval()
        
        epoch_IONs = [[] for _ in range(len(config.model.output_stages))] \
            if hasattr(config.model, "output_stages") else [[]]

        fine_epoch_IONS = []
        
        epoch_class_numbers = [torch.zeros_like(interval_values).to(device) for _ in range(len(config.model.output_stages))] \
            if hasattr(config.model, "output_stages") else [torch.zeros_like(interval_values).to(device)]
        fine_epoch_class_number = torch.zeros_like(interval_values).to(device)

        with torch.no_grad():
            for i,(imgs,target_maps,landmarks,multi_target_maps) in enumerate(val_loader):
                imgs = imgs.to(device)
                target_maps = target_maps.to(device)
                landmarks = landmarks.to(device)
                multi_target_maps = [stage_target_maps.to(device) for stage_target_maps in multi_target_maps]

                output = model(imgs)
                loss = cal_loss_fn(output, target_maps, multi_target_maps, criterion, model.training, resolution_weights)

                pred_landmarks,batch_IONs, batch_class_numbers = decode_heatmap_fn(output,landmarks,gt_heatmap_size)

                if is_val_roi and use_roi == 1:
                    pred_landmarks = pred_landmarks[0] # when only return 64 heatmaps
                    roi_feature = get_roi(config, output, pred_landmarks)
                    roi_offset = model.forward_roi(roi_feature)
                    fine_pred_landmarks = pred_landmarks + roi_offset
                    fine_batch_class_number, fine_batch_ION = cal_batch_ION(fine_pred_landmarks,landmarks)
                    fine_epoch_IONS += fine_batch_ION
                    fine_epoch_class_number += fine_batch_class_number
                    
                elif is_val_roi and use_roi > 1:
                    # return multiple heatmaps
                    max_resolution_pred_landmarks, concate_roi_features, resolution_landmark_offsets = get_multi_rois(config, output) # output is multi_stage_heatmaps when eval
                    roi_offset = model.forward_roi(concate_roi_features, resolution_landmark_offsets = resolution_landmark_offsets)
                    fine_pred_landmarks = max_resolution_pred_landmarks + roi_offset
                    fine_batch_class_number, fine_batch_ION = cal_batch_ION(fine_pred_landmarks,landmarks)
                    fine_epoch_IONS += fine_batch_ION
                    fine_epoch_class_number += fine_batch_class_number

                for stage_batch_IONs, stage_epoch_IONs, stage_epoch_class_number, stage_batch_class_number in zip(batch_IONs, epoch_IONs, epoch_class_numbers, batch_class_numbers):
                    stage_epoch_IONs += stage_batch_IONs # epoch_IONs += batch_IONs # list add  []+[1,2,3]+[2,3]=[1,2,3,2,3]
                    stage_epoch_class_number += stage_batch_class_number # eltwise add
                
                losses.update(loss.item(), imgs.size(0))

            nmes = []
            for stage_i in range(len(epoch_IONs)):
                
                res = config.heatmap.heatmap_size if not hasattr(model, "output_resolutions") else model.output_resolutions[stage_i]
                stage_epoch_IONs = epoch_IONs[stage_i]
                stage_epoch_IONs_np = np.array(stage_epoch_IONs)
                nme,fr,auc = compute_fr_and_auc(stage_epoch_IONs_np) 
                nme,fr_08,auc_08 = compute_fr_and_auc(stage_epoch_IONs_np, thres=0.08)
                logger.info(f"Val Info: {res}x{res} | nme:{nme:.5f}, FR_10:{fr:.5f}, AUC_10:{auc:.5f}, FR_08:{fr_08:.5f}, AUC_08:{auc_08:.5f} | Time cost: {(time.time()-start_time):.2f}s \n")
                nmes.append(nme) # save stage nme for different resolutions

                stage_epoch_class_number = epoch_class_numbers[stage_i]
                stage_epoch_class_ratio_list = (stage_epoch_class_number / stage_epoch_class_number.sum()).cpu().tolist()
                interval_values_list = interval_values.cpu().tolist()
                interval_cls_str = ""
                for interval, cls_value in zip(interval_values_list, stage_epoch_class_ratio_list):
                    interval_cls_str += f"{interval:.3f} : {cls_value:.3f} | "
                logger.info(f"{res}x{res} nme: {nme:.5f} | {interval_cls_str} \n")
                

            if is_val_roi and use_roi:
                fine_epoch_IONs_np = np.array(fine_epoch_IONS)
                nme,fr,auc = compute_fr_and_auc(fine_epoch_IONs_np) 
                nme,fr_08,auc_08 = compute_fr_and_auc(fine_epoch_IONs_np, thres=0.08)
                logger.info(f"Val Info: Fine (ROI FC) | nme:{nme:.5f}, FR_10:{fr:.5f}, AUC_10:{auc:.5f}, FR_08:{fr_08:.5f}, AUC_08:{auc_08:.5f} | Time cost: {(time.time()-start_time):.2f}s \n")
                nmes.insert(0, nme)
                
                fine_class_ratio_list = (fine_epoch_class_number / fine_epoch_class_number.sum()).cpu().tolist()
                interval_values_list = interval_values.cpu().tolist()
                interval_cls_str = ""
                for interval, cls_value in zip(interval_values_list, fine_class_ratio_list):
                    interval_cls_str += f"{interval:.3f} : {cls_value:.3f} | "
                logger.info(f"Final ROI_FC nme: {nme:.5f} | {interval_cls_str} \n")

            return losses.avg, nmes[0]

def get_decode_fn(config):
    if config.model.backbone in ["hourglass"]:
        if hasattr(config.model, "use_roi") and config.model.use_roi > 1:
            return decode_hourglass_roi_output
        else:
            return decode_hourglass_output
    elif config.model.backbone in ["resnet18", "resnet34", "resnet50"]:
        return decode_resnet_fc_output
    else:
        raise ValueError("No suitable function now")

def decode_hourglass_roi_output(output, gt_landmarks, heatmap_size=64):
    return decode_resnet_fc_output(output, gt_landmarks,heatmap_size)

def decode_hourglass_output(output, gt_landmarks, heatmap_size=64):
    # 因为hourglass 在eval模式下仅输出最后一级的heatmap
    pred_landmarks = decode_head(output)
    class_number, batch_IONs = cal_batch_ION(pred_landmarks,gt_landmarks)
    return [pred_landmarks], [batch_IONs], [class_number]

def decode_resnet_fc_output(output, gt_landmarks, heatmap_size=64):
    # heatmap size describe the resolution for gt_landmarks, default is 64

    pred_landmarks, batch_IONs, class_numbers = [], [], []
    for stage_output in output:
        stage_pred_landmarks = decode_head(stage_output)
        stage_ratio = heatmap_size / stage_output.size(-1) 
        stage_gt_landmarks = gt_landmarks / stage_ratio
        class_number, stage_batch_IONs = cal_batch_ION(stage_pred_landmarks, stage_gt_landmarks)
        pred_landmarks.append(stage_pred_landmarks)
        batch_IONs.append(stage_batch_IONs)
        class_numbers.append(class_number)

    return pred_landmarks, batch_IONs, class_numbers


def test_model(test_info, logger):
    pass

def early_stop(nme, best_res, epoch, config):
    if epoch > 10 and nme > 0.5:
        print(f"Not convergence! at epoch {epoch}")
        return True
    
    if config.train.scheduler == "ReduceLROnPlateau" and epoch > best_res['epoch'] + 10:
        return True

    if config.train.scheduler == "MultiStepLR" and epoch > config.train.decay_steps[-1] + 5 and epoch > best_res['epoch'] + 10:
        return True

    return False
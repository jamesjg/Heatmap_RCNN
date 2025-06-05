import torch
from torch import nn
from lib.loss import *
import numpy as np
from scipy.integrate import simps


def lr_repr(optim):
    _lr_repr_ = ''
    for pg in optim.param_groups:
        _lr_repr_ += ' {} '.format(pg['lr'])
    return _lr_repr_


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_single_loss(config,criterion_str):
    if criterion_str in ['L2','MSE','REG']:
        criterion = nn.MSELoss()
    elif criterion_str == 'L1':
        criterion = nn.L1Loss()
    elif criterion_str == 'SMOOTH_L1':
        criterion = nn.SmoothL1Loss()
    elif criterion_str == 'CLS':
        criterion = CE_Loss(int(config.heatmap.heatmap_size*config.heatmap.heatmap_size))
    elif criterion_str == 'KL':
        criterion = SoftKLLoss()
    elif criterion_str == 'MIX_L2_CE':
        criterion = MixLoss_L2ANDCEL(config.model.loss_intern_alpha, int(config.heatmap.heatmap_size*config.heatmap.heatmap_size))
    elif criterion_str == 'MIX_AWING_CE':
        criterion = MixLoss_AWINGANDCEL(config.model.loss_intern_alpha, int(config.heatmap.heatmap_size*config.heatmap.heatmap_size))
    elif criterion_str == 'MIX_L2_KL':
        criterion = MixLoss_L2ANDKL(config.model.loss_intern_alpha)
    elif criterion_str in ["AWING", "ADAPTIVE_WING"]:
        criterion = AdaptiveWingLoss()
    else:
        raise ValueError("Not support {} loss now.".format(criterion_str))
    return criterion

def get_loss(config):
    criterion = dict()
    assert hasattr(config.model,"loss") , "Not exist criterion_heatmap in config file"

    criterion.update({
        "loss": get_single_loss(config, config.model.loss.upper())
    })

    if hasattr(config.model, "sup_losses") :
        sup_losses = []
        for stage_loss in config.model.sup_losses:
            sup_losses.append(get_single_loss(config, stage_loss))
        criterion.update({
            "sup_losses":sup_losses
        })
    
    if hasattr(config.model, "roi_loss"):
        criterion.update({
            "roi_loss":get_single_loss(config, config.model.roi_loss.upper())
        })

    return criterion

    if hasattr(config.model, "sup_losses") and hasattr(config.model, "loss") and hasattr(config.model, "roi_loss"):
        sup_losses = []
        for stage_loss in config.model.sup_losses:
            sup_losses.append(get_single_loss(config, stage_loss))
        main_loss = get_single_loss(config, config.model.loss.upper())
        roi_loss = get_single_loss(config, config.model.roi_loss.upper())

        return dict(
            sup_losses = sup_losses,
            loss = main_loss,
            roi_loss=roi_loss,
        )
    
    if config.model.loss is not None:
        criterion_str = config.model.loss.upper()
        return get_single_loss(config, criterion_str)
    
    
def compute_fr_and_auc(nmes, thres=0.10, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    nme = np.mean(nmes)

    # print("NME %: {}".format(np.mean(nmes)*100))
    # print("FR_{}% : {}".format(thres,fr*100))
    # print("AUC_{}: {}".format(thres,auc))
    return nme, fr, auc


def class_loss():

    # the first index is zero placeholder, and last represent means > 3
    class_intervals_square = torch.tensor([100,1,2,4,5,8,9]) #,10,13,16,17
    class_intervals = torch.sqrt(class_intervals_square)
    class_intervals[0] = torch.tensor(0.)
    
    return class_intervals

interval_values = class_loss()

def cal_hist_cls_error(landmark_diffs):
    """
    calculate the hist on class intervals

        Args: landmark_diffs: batchx98x2, it equals pred_landmarks - gt_landmarks
        return:
            class_numbers: shape is [11], the number for the intervals,  Note the class_numbers.sum() equals to batch*98

    """
    # landmark_diffs = pred_landmarks - gt_landmarks, shape is (32,98,2) 
    loss_intervals = interval_values.to(landmark_diffs.device)

    distance = torch.norm(landmark_diffs,dim=-1).view(-1,1) # (32,98) 先计算距离
    distance_exp = distance.expand(distance.shape[0], loss_intervals.shape[0]) # expand 之后可以用矩阵直接减
    abs_class = (distance_exp - loss_intervals).abs() # 算一下到interval最小距离
    min_error_index = torch.argmin(abs_class, -1) # 属于距离最小的interval
    class_numbers = []
    for i in range(loss_intervals.shape[0]):
        cls_number = (min_error_index == i).sum() # 统计
        class_numbers.append(cls_number)

    class_numbers = torch.tensor(class_numbers).to(landmark_diffs.device)

    return class_numbers

def cal_dist_on_interval_label(landmark_pred, landmark_gt):
    landmark_diffs = landmark_gt - landmark_pred
    distance_interval_labels = torch.tensor([0,1.0,1.414,2.0,2.236,2.828,3]).to(landmark_pred.device)
    error_distance = torch.norm(landmark_diffs,dim=-1).view(-1,1) # calculate the error distance between pred and gt
    error_distance_exp = error_distance.expand(error_distance.shape[0], distance_interval_labels.shape[0])
    abs_distance_error_label = (error_distance_exp - distance_interval_labels).abs() # calculate the loss between error label and error distance
    min_error_label_index = torch.argmin(abs_distance_error_label, -1) # find the nearest label
    
    # calculate the histogram distribution
    hist_dist = []
    for i in range(distance_interval_labels.shape[0]):
        label_dist = (min_error_label_index == i).sum()
        hist_dist.append(label_dist)
    hist_dist = torch.tensor(hist_dist).to(landmark_diffs.device)

    # calculate the cumulative probability distribution
    cum_dist = hist_dist.clone()
    for i in range(1,distance_interval_labels.shape[0]):
        cum_dist[i] += cum_dist[i-1]

    return hist_dist, cum_dist

def cal_batch_ION(pred_landmarks, gt_landmarks):
   
    # (n,98,2)
    norm_indices = None  # normalization factor (distance) between which points
    num_landmarks = gt_landmarks.size(1)
    if num_landmarks == 68:
        norm_indices = [36,45]
    elif num_landmarks == 29:
        norm_indices = [8,9]
        # norm_indices = [17,16] # IPN
    elif num_landmarks == 98:
        norm_indices = [60,72]
    else:
        print("No such data!")
        exit(0)
    
    batch_IONs = []

    landmark_diffs = pred_landmarks - gt_landmarks
    # class_numbers, cum_numbers = cal_dist_on_interval_label(pred_landmarks, gt_landmarks)
    class_numbers = cal_hist_cls_error(landmark_diffs)
    for landmark_diff, gt_landmark in zip(landmark_diffs, gt_landmarks):
        norm_distance = torch.norm(gt_landmark[norm_indices[0]] - gt_landmark[norm_indices[1]])
        ION = torch.sum(torch.norm(landmark_diff,dim=1)) / (num_landmarks * norm_distance)
        batch_IONs.append(ION.item())
    
    return class_numbers, batch_IONs

def need_val(epoch,config):
    val_info = np.array(config.train.val_epoch)
    val_epochs = val_info[:,0]
    val_steps = val_info[:,1]

    for end_epoch,step in zip(val_epochs,val_steps):
        if epoch < end_epoch:
            if epoch % step == 0:
                return True
            else:
                return False
        
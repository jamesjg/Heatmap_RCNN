from math import ceil
import torch.utils.data as data
import torch
import numpy as np
import sys
sys.path.append('.')

"""
    the landmarks output from all head methods should be correspond to the heatmap size.
"""

# Gauss heatmap should change value TODO
def gen_heat(sigma=0.):

    if sigma == 0 :
        return np.array([[1]])

    tmp_size = sigma * 3
    size = 2 * tmp_size + 1
    x = np.arange(0,size,1,dtype=np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # print(f"Generate Gauss heatmap | {g.shape}")
    return g

def gen_multi_heats(multi_stage_sigmas):
    
    multi_heats = []
    for stage_sigma in multi_stage_sigmas:
        multi_heats.append(gen_heat(stage_sigma))
    return multi_heats


def encode_head(target_w_size,heat,heatmap_size,heatmap_sigma,config):
    # without offset map head: normal head
    target_map = np.zeros((config.data.num_landmarks,heatmap_size,heatmap_size),dtype=np.float32)

    # if config.heatmap.heatmap_method == "GAUSS":
    heat = heat.copy()
    tmp_size = heatmap_sigma * 3
    # judge the kernel is odd
    is_kernel_int = True if abs(round(tmp_size) - tmp_size) < 0.11111 else False  
    
    if heat.size == 1:
        clip_target_w_size = np.clip(target_w_size, 0, heatmap_size-1)
        for i in range(clip_target_w_size.shape[0]):
            pt,pt_map = clip_target_w_size[i],target_map[i]
            pt_map[round(pt[1]+ 1e-7), round(pt[0]+ 1e-7)] = 1.0
            
        return target_map

    for i in range(target_w_size.shape[0]):
        # pt : [x,y], pt_map : [heatmap_size x heatmap_size]
        pt,pt_map = target_w_size[i],target_map[i]
        if is_kernel_int:
            ul =  np.array([round(pt[0] - tmp_size + 1e-7), round(pt[1] - tmp_size + 1e-7)],dtype=int)
            br =  np.array([round(pt[0] + tmp_size + 1e-7), round(pt[1] + tmp_size + 1e-7)],dtype=int)
        else:
            # floor
            ul = np.array([pt[0] - tmp_size, pt[1] - tmp_size],dtype=int)
            br = np.array([pt[0] + tmp_size, pt[1] + tmp_size],dtype=int)

        # Check that any part of the gaussian is in-bounds
        if (ul[0] >= pt_map.shape[1] or ul[1] >= pt_map.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            continue
        br += 1 # take the same process as shape, thus add 1
        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], pt_map.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], pt_map.shape[0]) - ul[1]
        # Image range
        pt_x = max(0, ul[0]), min(br[0], pt_map.shape[1])
        pt_y = max(0, ul[1]), min(br[1], pt_map.shape[0])

        pt_map[pt_y[0]:pt_y[1], pt_x[0]:pt_x[1]] = heat[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target_map


def gen_multi_res(target, config, multi_heats, multi_heatmap_sizes):
    """
        Args:
            target: (98,2)  in [0,1)
            config: dict, reference on the resnet_l2.py
        
        return multiple stage heatmaps
    """
        
    multi_heatmaps = []
    multi_target_w_size = [target * stage_heatmap_size for stage_heatmap_size in multi_heatmap_sizes]
    
    for target_w_size, heat, heatmap_size, heatmap_sigma in zip(multi_target_w_size, multi_heats,multi_heatmap_sizes, config.model.multi_stage_sigmas):
        multi_heatmaps.append(encode_head(target_w_size, heat, heatmap_size, heatmap_sigma, config))

    return multi_heatmaps

def decode_head(target_maps):
    """
        Args:
            target_maps (n,98,64,64) tensor float32

        return : 
            preds (n,98,2)
    """
    max_v,idx = torch.max(target_maps.view(target_maps.size(0),target_maps.size(1),target_maps.size(2)*target_maps.size(3)), 2)
    preds = idx.view(idx.size(0),idx.size(1),1).repeat(1,1,2).float()
    max_v = max_v.view(idx.size(0),idx.size(1),1)
    pred_mask = max_v.gt(0).repeat(1, 1, 2).float()

    preds[..., 0].remainder_(target_maps.size(3))
    preds[..., 1].div_(target_maps.size(2)).floor_()

    """
    # add gradients in dx and dy, but get a little effects.
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = target_maps[i, j, :]
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            if pX > 0 and pX < target_maps.size(3) - 1 and pY > 0 and pY < target_maps.size(2) - 1:
                # make sure the border points' neighbors are in the heatmaps
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]]).to(preds)
                preds[i, j].add_(diff.sign_().mul_(.25))
    """
    preds.mul_(pred_mask)
    return preds


if __name__ == "__main__":
    from lib.utils.parser import train_parser_args as parser_args
    import mmcv
    from lib.utils.config_tool import merge_args_into_config
    args, unparsed = parser_args()
    config = mmcv.Config.fromfile(args.config_file)
    merge_args_into_config(args, unparsed, config)

    norm_indices = None
    if config.data.data_type == "300W":
        norm_indices = [36,45]
    elif config.data.data_type == "COFW":
        norm_indices = [8,9]
    elif config.data.data_type == "WFLW":
        norm_indices = [60,72]
    elif config.data.data_type == "AFLW":
        pass
    else:
        print("No such data!")
        exit(0)
    
    from PIL import Image
    import os
    import time
    img = Image.open("data/benchmark/WFLW/test_with_box/wflw_test_with_box_1.jpg").convert('RGB') 
    heat = gen_heat(config.heatmap.heatmap_sigma)

    root_folder = os.path.join("data","benchmark")
    label_path = os.path.join(root_folder,config.data.data_type,config.data.data_folder+".txt")
    with open(label_path,'r') as f:
        data_txt = f.readlines()
    data_info = np.array([x.strip().split() for x in data_txt])
    pts_array = data_info[:,1:].astype(np.float32).reshape(data_info.shape[0],-1,2).copy()
    ION = []
    gen_times = []
    decode_times = []

    from lib.dataset.augmentation import pad_crop
    from torch.nn import functional as F
    for target in pts_array:

        start_time = time.time()
        _,target = pad_crop(img,target)

        target = target * config.heatmap.heatmap_size
        target_w_size = target.copy()
        target_map  = encode_head(target_w_size,heat,config.heatmap.heatmap_size, config.heatmap.heatmap_sigma, config)
        target_maps = torch.unsqueeze(torch.from_numpy(target_map).float(),dim=0)
        gen_time = time.time()

        # if config.head_type == 'cls':
        #     if len(target_maps.shape) == 2:
        #         target_maps = torch.clip(target_maps,0,config.heatmap_size * config.heatmap_size -1)
        #         target_maps = F.one_hot(target_maps.to(torch.int64),config.heatmap_size * config.heatmap_size)

        preds = decode_head(target_maps)
        decode_time = time.time()
        
        # print(len(preds[(preds.int() - target_w_size.astype('int')).abs() > 0]))

        diff = target - torch.squeeze(preds).numpy()
        norm = np.linalg.norm(target[norm_indices[0]] - target[norm_indices[1]]) if norm_indices is not None else config.heatmap.heatmap_size
        c_ION = np.sum(np.linalg.norm(diff,axis=1))/(diff.shape[0]*norm)
        ION.append(c_ION)

        gen_times.append(gen_time - start_time)
        decode_times.append(decode_time - gen_time)
        # print(np.sum(np.linalg.norm(diff,axis=1))/(diff.shape[0]*norm))
    M_ION = np.mean(ION)
    print(M_ION)
    print(np.mean(gen_times),np.mean(decode_times))

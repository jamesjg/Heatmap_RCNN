from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import sys

from lib.dataset.head import encode_head,gen_heat, gen_multi_res,gen_multi_heats
sys.path.append('.')
from lib.dataset.augmentation import *

class FA_dataset(Dataset):
    def __init__(self,config,read_mem=True,Is_train=True,transform=None):
        """
            config : experiments/**
            read_mem : bool, read dataset in memery or not.
            Is_train : bool, train or test
            transform: default is None, operations in torchvision
        """
        root_folder = os.path.join("data","benchmark")
        self.data_folder = config.data.data_folder if Is_train else config.data.test_folder
        self.data_path = os.path.join(root_folder,config.data.data_type,self.data_folder)
        self.points_flip = flip_points(config.data.data_type)
        self.is_train = Is_train
        self.transform = transform
        self.read_mem = read_mem
        self.config = config
        
        self.heat = gen_heat(config.heatmap.heatmap_sigma)
        self.multi_stage_heats = gen_multi_heats(config.model.multi_stage_sigmas) if hasattr(config.model, "multi_stage_sigmas") else []
        self.multi_stage_resolutions = [int(config.heatmap.heatmap_size / ( 2 ** (r-1) )) for r in config.model.output_stages] if hasattr(config.model, "output_stages") else []

        label_path = os.path.join(root_folder,config.data.data_type,self.data_folder+".txt")
        with open(label_path,'r') as f:
            data_txt = f.readlines()
        data_info = np.array([x.strip().split() for x in data_txt])
        
        self.img_paths = data_info[:,0].copy()
        self.pts_array = data_info[:,1:].astype(np.float32).reshape(data_info.shape[0],-1,2).copy()
        self.imgs = [Image.open(os.path.join(self.data_path,img_path)).convert('RGB') 
                    for img_path in self.img_paths] if read_mem else []
        # check image size
        check_size(self.imgs,config)
        print("Finish READ and CHECK dataset, Success !")

    def __getitem__(self,index):

        img = self.imgs[index].copy() if self.read_mem \
              else Image.open(os.path.join(self.data_path,self.img_paths[index])).convert('RGB')
        target = self.pts_array[index].copy()

        if self.is_train:
            img, target = random_translate(img,target)
            img, target = random_flip(img, target, self.points_flip)
            img, target = random_rotate(img, target, angle_max=30)
            img = random_blur(img)
            img = random_occlusion(img)

        # ignore or pad crop,both are ok.
        img, target = pad_crop(img,target)
        # target = ignore_crop(target)  

        target_w_size = target * self.config.heatmap.heatmap_size
        target_map = encode_head(target_w_size,self.heat,self.config.heatmap.heatmap_size,self.config.heatmap.heatmap_sigma,self.config)
        
        multi_target_maps = []
        if hasattr(self.config.model, "output_stages"):
            multi_target_maps = gen_multi_res(target, self.config, self.multi_stage_heats, self.multi_stage_resolutions)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        target_w_size = torch.from_numpy(target_w_size).float()
        target_map = torch.from_numpy(target_map).float()

        return img,target_map,target_w_size, multi_target_maps

    def __len__(self):
        return self.pts_array.shape[0]

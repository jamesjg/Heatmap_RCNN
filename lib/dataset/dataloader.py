from torchvision import transforms
import sys
sys.path.append('.')
from lib.dataset.dataset import FA_dataset
from torch.utils.data import DataLoader

# transformation
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomGrayscale(0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # return None, None # Debug with visualization
    return train_transform, val_transform


# dataset
def get_datasets(config):
    train_trans, val_trans = get_transforms()
    train_dataset = FA_dataset(config, read_mem=True, Is_train=True, transform=train_trans)
    val_dataset = FA_dataset(config, read_mem=False, Is_train=False, transform=val_trans)
    return train_dataset, val_dataset

def get_train_dataset(config):
    train_trans, _ = get_transforms()
    train_dataset = FA_dataset(config, read_mem=True, Is_train=True, transform=train_trans)
    return train_dataset

def get_val_dataset(config):
    _, val_trans = get_transforms()
    val_dataset = FA_dataset(config, read_mem=False, Is_train=False, transform=val_trans)
    return val_dataset


# dataloader
def get_dataloaders(config):
    train_dataset, val_dataset = get_datasets(config)
    train_loader = DataLoader(
                    train_dataset, 
                    batch_size=config.train.batch_size,
                    shuffle=True,
                    num_workers=config.train.num_workers,
                    drop_last=False,
                    pin_memory=True)
    
    val_loader = DataLoader(
                    val_dataset, 
                    batch_size=config.train.batch_size,
                    shuffle=False,
                    num_workers=config.train.num_workers,
                    drop_last=False,
                    pin_memory=True)

    return train_loader, val_loader

def get_train_dataloader(config):
    train_dataset = get_train_dataset(config)
    train_loader = DataLoader(
                    train_dataset, 
                    batch_size=config.train.batch_size,
                    shuffle=True,
                    num_workers=config.train.num_workers,
                    drop_last=False,
                    pin_memory=True)
    
    return train_loader

def get_val_dataloader(config):
    val_dataset = get_val_dataset(config)
    val_loader = DataLoader(
                    val_dataset, 
                    batch_size=config.train.batch_size,
                    shuffle=False,
                    num_workers=config.train.num_workers,
                    drop_last=False,
                    pin_memory=False)
    
    return val_loader


if __name__ == "__main__":
    from lib.utils.parser import train_parser_args as parser_args
    import mmcv
    from lib.utils.config_tool import merge_args_into_config
    args, unparsed = parser_args()
    config = mmcv.Config.fromfile(args.config_file)
    merge_args_into_config(args, unparsed, config)

    train_loader, val_loader = get_dataloaders(config)

    import matplotlib.pyplot as plt
    import os
    import numpy as np
    for i,(imgs,target_maps,landmarks) in enumerate(train_loader):
        landmarks = landmarks.numpy().astype(np.float32).reshape(imgs.shape[0], -1,2)
        img_landmarks = imgs.shape[2] * landmarks / config.heatmap.heatmap_size
        for j in range(imgs.size(0)):
            img = transforms.ToPILImage()(imgs[j]).convert("RGB")
            landmark = img_landmarks[j]
            plt.imshow(img)
            plt.scatter(landmark[:,0],landmark[:,1],1)
            # plt.scatter(10,50,1)
            plt.savefig(os.path.join("images","{}_{}.jpg".format(i,j)))
            plt.close()
            print(i,j)
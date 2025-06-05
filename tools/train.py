import os
import sys
sys.path.append('.')

import torch
import mmcv

from lib.utils.parser import parser_args
from lib.utils.config_tool import create_logger,create_exp_dir, merge_args_into_config,seed_everything
from lib.utils.model import get_model, train_model, val_model,early_stop, finetune_roi
from lib.utils.loss import get_loss,lr_repr, need_val
from lib.utils.scheduler import Scheduler, get_optim
from lib.dataset.dataloader import get_dataloaders
from lib.utils.saver import load_ckpt, save_ckpt

def train(config:mmcv.Config):

    # create model folder and logger
    create_exp_dir(config)
    logger = create_logger(config)
    config.dump(os.path.join(config.exp_dir, 'config.py'))
    
    # set gpu and random seed
    device = torch.device(f'cuda:{config.train.gpu_id}')
    seed_everything(config.train.seed)

    # init model
    model = get_model(config).to(device)
    if args.init_roi_weight:
        # 相当于只考虑backbone, roi_module在finetune时随机初始化
        model.roi_module.init_weights()

    optimizer = get_optim(config, model)

    # load_resume_checkpoint
    best_res = load_ckpt(config, model)
    logger.info(f"Init Result: {best_res}")

    # init loss and scheduler
    criterion = get_loss(config)
    scheduler = Scheduler(config, optimizer)

    # init dataset 
    train_loader, val_loader = get_dataloaders(config)

    train_info = {
        "config":config,
        "model":model,
        "optimizer":optimizer,
        "criterion":criterion,
        "scheduler":scheduler,
        "dataloader":train_loader,
        "device":device,
        "epoch":0,
        "train_backbone":args.train_backbone, # only train backbone (the part which output heatmaps)
    }

    val_info = {
        "config":config,
        "model":model,
        "criterion":criterion,
        "dataloader":val_loader,
        "best_res":best_res,
        "device":device,
        "epoch":0,
        "val_backbone":args.train_backbone, # only val backbone (the part which output heatmaps)
    }

    for epoch in range(config.train.num_epochs):

        train_info.update({"epoch": epoch})
        val_info.update({"epoch": epoch})

        # train 
        logger.info(f"\n----------------------Step into {epoch} epoch, Lr:{lr_repr(optimizer)}----------------------")
        train_loss = train_model(train_info, logger) if not args.finetune else finetune_roi(train_info,logger)

        # validation
        val_loss, nme = val_model(val_info,logger)
        if nme < best_res["nme"]:
            best_res.update({"epoch":epoch,"loss":val_loss,"nme":nme})
            logger.info(f"Update best res and save ckpt: {best_res}")
            save_ckpt(val_info)

        scheduler.step(metrics=nme)

        if early_stop(nme,best_res,epoch,config):
            logger.info("Early stop training....")
            return best_res
    
    return best_res


if __name__ == "__main__":
    # process args and config
    args, unparsed = parser_args()
    config = mmcv.Config.fromfile(args.config_file)
    merge_args_into_config(args, unparsed, config)

    best_res = train(config)

    print(best_res)
    print("Done")

import os
import sys
sys.path.append('.')

import torch
import mmcv

from lib.utils.parser import parser_args
from lib.utils.config_tool import create_logger,create_exp_dir, merge_args_into_config,seed_everything
from lib.utils.model import get_model, val_model
from lib.utils.loss import get_loss
from lib.dataset.dataloader import get_val_dataloader
from lib.utils.saver import load_ckpt

# 300W 
label_300W = [
        "valid",
        "valid_common",
        "valid_challenge",
        "test",
]

# TODO
# label_300W_GT = [
#         "valid",
#         "valid_common",
#         "valid_challenge",
#         "test",
# ]

label_COFW = [
    "test"
]

# WFLW
label_WFLW = [
        "test",
        "test_largepose",
        "test_expression",
        "test_illumination",
        "test_makeup",
        "test_occlusion",
        "test_blur"
]

def test(config:mmcv.Config):

    # create model folder and logger
    create_exp_dir(config,Is_test=True)
    logger = create_logger(config, Is_test=True)
    config.dump(os.path.join(config.exp_dir, 'config.py'))
    
    # set gpu and random seed
    device = torch.device(f'cuda:{config.train.gpu_id}')
    seed_everything(config.train.seed)

    # init model
    model = get_model(config).to(device)

    # load_resume_checkpoint
    best_res = load_ckpt(config, model)
    logger.info(f"Init Result: {best_res}")

    # init loss and scheduler
    criterion = get_loss(config)
    if args.test_all:
        test_datasets = eval(f"label_{config.data.data_type}")
    else:
        test_datasets = [eval(f"label_{config.data.data_type}")[0]]
    
    for test_subset in test_datasets:
        config.data.test_folder = test_subset
        logger.info(f"\n\n\nTest NME for {config.data.data_type} : {test_subset}")
        # init dataset 
        val_loader = get_val_dataloader(config)

        val_info = {
            "config":config,
            "model":model,
            "criterion":criterion,
            "dataloader":val_loader,
            "best_res":best_res,
            "device":device,
            "val_backbone":args.train_backbone, # only val backbone (the part which output heatmaps)
        }

        # validation
        val_loss, nme = val_model(val_info,logger)
    
    return nme


if __name__ == "__main__":
    # process args and config
    args, unparsed = parser_args()
    config = mmcv.Config.fromfile(args.config_file)
    merge_args_into_config(args, unparsed, config)

    best_res = test(config)

    print(best_res)
    print("Done")

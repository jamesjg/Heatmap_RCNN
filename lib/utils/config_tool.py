import logging
import os
import sys

sys.path.append('.')
import numpy as np
import random
from datetime import datetime
import mmcv
import torch

def create_exp_dir(config, Is_test=False):
    log_folder = 'logs' if not Is_test else 'test_logs'

    model_folder_path = os.path.join(log_folder, config.data.data_type.upper(), config.model.backbone,config.exp_name)

    version_ = 0
    if os.path.exists(model_folder_path):
        version_ = len(os.listdir(model_folder_path))
    
    model_folder_path = os.path.join(model_folder_path, str(version_))

    os.makedirs(model_folder_path)

    print(f"Create folder : {model_folder_path} Success!")

    setattr(config,'exp_dir',model_folder_path)


def parse_str_to_array(array_str):
    # example (Bool, Int, Float, String)
    # 0.01,0.02,0.03 -> [0,01,0.02,0.03]
    # 1,2,3 -> [1,2,3]
    # REG,CLS -> ["REG", CLS]

    array_str_list = array_str.split(',')
    val_array = []
    for val_str in array_str_list:
        val_res = None
        if val_str.upper() == "TRUE":
            val_res = True
        elif val_str.upper() == "FALSE":
            val_res = False
        elif '.' in val_str or 'e' in val_str: # float
            try:
                val_res = float(val_str)
            except:
                pass
        else: # int
            try:
                val_res = int(val_str)
            except:
                print("WARN: Cannot regcongnize the data type of {} ! Process as string type".format(val_str))
                val_res = val_str
        
        val_array.append(val_res)
    
    return val_array

def merge_args_into_config(args, unparsed, config:mmcv.Config):
    
    # merge parsed args
    if args.model_dir is None:
        args.model_dir = datetime.now().strftime("%m_%d_%H_%M_%S")
    config.__setattr__('exp_name', args.model_dir)

    if args.ckpt is not None:
        config.model.__setattr__('ckpt', args.ckpt)
    
    # merge unparsed args
    for up_kv in unparsed:
        up_kv = up_kv.replace('--','').split(':')
        up_kv = up_kv[0].split('=') if len(up_kv) == 1 else up_kv
        assert len(up_kv) == 2 , "unparsed item only support ':' or '=', like --batch_size=8"

        up_key = up_kv[0]
        up_val = up_kv[1]
        
        # judge type: bool, float, int
        if ',' in up_val:
            up_val = parse_str_to_array(up_val)
        elif up_val.upper() == "TRUE":
            up_val = True
        elif up_val.upper() == "FALSE":
            up_val = False
        elif '.' in up_val or 'e' in up_val: # float
            try:
                up_val = float(up_val)
            except:
                pass
        else: # int
            try:
                up_val = int(up_val)
            except:
                print("WARN: Cannot regcongnize the data type of {} ! Process as string type".format(up_val))
        
        if up_key in config.data.keys():
            print(f"Change data.{up_key} from {config.data[up_key]} to {up_val}")
            config.data.__setattr__(up_key,up_val)
        elif up_key in config.heatmap.keys():
            print(f"Change heatmap.{up_key} from {config.heatmap[up_key]} to {up_val}")
            config.heatmap.__setattr__(up_key,up_val)
        elif up_key in config.model.keys():
            print(f"Change model.{up_key} from {config.model[up_key]} to {up_val}")
            config.model.__setattr__(up_key,up_val)
        elif up_key in config.backbone.keys():
            print(f"Change backbone.{up_key} from {config.backbone[up_key]} to {up_val}")
            config.backbone.__setattr__(up_key,up_val)
        elif up_key in config.train.keys():
            print(f"Change train.{up_key} from {config.train[up_key]} to {up_val}")
            config.train.__setattr__(up_key,up_val)
        else:
            assert hasattr(config,up_key), "config file unsupport the item: {}".format(up_key)


def update_config_from_ckpt(ckpt_config, input_config):
    pass


def create_logger(config, Is_test=False):

    # log_file = "{}_{}_{}_{}_{}_{}.log".format(config.data_type.upper(),config.head_type.upper(),config.model_dir,config.heatmap_size,config.heatmap_method,config.criterion_heatmap)

    log_file = f"train.log" if not Is_test else "test.log"
    final_log_file = os.path.join(config.exp_dir, log_file)

    if os.path.exists(final_log_file):
        print("Current log file is exist")
        raise ValueError("Log file alread exist")

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger


def seed_everything(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

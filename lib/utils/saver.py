import os
import torch
import math

def save_ckpt(val_info, optimizer=None):
    # save only one ckpt repeatly in the model dir
    # the model contains the config, best res, model_dict
    config = val_info["config"]
    best_res = val_info["best_res"]
    model_state_dict = val_info["model"].state_dict()
    ckpt_path = os.path.join(config.exp_dir,"ckpt.pth")

    save_info = {
        "model_state_dict":model_state_dict,
        "best_res":best_res,
        "config":config
    }
    if optimizer is not None:
        save_info.update({"optimizer_state_dict":optimizer.state_dict()})
    
    torch.save(save_info, ckpt_path)


def load_ckpt(config, model, optimizer=None):
    # Init best result
    best_res = {'epoch':0,'nme':math.inf,'loss': math.inf ,'pupil_nme':None}
    if config.model.ckpt is None or not os.path.exists(config.model.ckpt):
        return best_res

    save_info = torch.load(config.model.ckpt, map_location=(f'cuda:{config.train.gpu_id}'))
    save_config = save_info["config"]
    save_config.dump(os.path.join(config.exp_dir, 'load_ckpt_config.py'))

    best_res.update(save_info["best_res"])

    model.load_state_dict(save_info["model_state_dict"], strict=True)

    if optimizer is not None and hasattr(save_info ,"optimizer_state_dict"):
        optimizer.load_state_dict(save_info["model_state_dict"])
    
    return best_res

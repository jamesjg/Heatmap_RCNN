import torch
from torch import optim

def get_optim(config,model):
    optimizer = None
    # if True:
    if hasattr(config.train, "roi_init_lr"):
        hourglass_params = [p for name, p in model.named_parameters() if not name.startswith('roi_module') and p.requires_grad]
        roi_params = [p for name, p in model.named_parameters() if name.startswith('roi_module') and p.requires_grad]
        optimizer = optim.Adam([
            {'params':hourglass_params, 'lr':config.train.init_lr, 'weight_decay':5e-4},
            {'params':roi_params, 'lr':config.train.roi_init_lr, 'weight_decay':5e-4}
        ])
    else:
        optimizer = optim.Adam(model.parameters(),lr=config.train.init_lr,weight_decay=5e-4)

    return optimizer

def get_scheduler(config, optimizer):

    scheduler = None
    if config.train.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.2,patience=2,verbose=True,min_lr=1e-8)
    elif config.train.scheduler == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.decay_steps, gamma=0.1)
    elif config.train.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=config.train.decay_steps, gamma=0.1)
    else:
        raise ValueError(f"Not Support scheduler for {config.train.scheduler}")
    
    return scheduler


class Scheduler():
    def __init__(self, config, optimizer) -> None:
        self.config = config
        self.scheduler = get_scheduler(config, optimizer)
            
    def step(self, metrics=None, epoch=None):
        if self.config.train.scheduler == "ReduceLROnPlateau":
            self.scheduler.step(metrics=metrics)
        else:
            self.scheduler.step()
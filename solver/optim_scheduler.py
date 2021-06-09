from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR


def get_optimizer(cfg, model_params):
    optim_name = cfg.SOLVER.optim
    if optim_name == 'adam':
        return Adam(model_params, lr=cfg.SOLVER.lr)
    elif optim_name == 'sgd':
        return SGD(model_params, lr=cfg.SOLVER.lr, momentum=0.9)


def get_lr_scheduler(cfg, optim=None):
    sche_name = cfg.SOLVER.scheduler
    if sche_name == 'steplr':
        return StepLR(optim, 50, gamma=0.1)
    elif sche_name == 'MultiStep':
        return MultiStepLR(optim, cfg.SOLVER.lr_step, gamma=0.1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lr_scheduler
from optimizer.radam import RAdam, AdamW

class Optimizer(nn.Module):
    def __init__(self, model, epoch):
        super(Optimizer, self).__init__()
        self.last_epoch = epoch
        self.setup_optimizer(model)

    def setup_optimizer(self, model):

        if 'Lambda' in cfg.SOLVER.LR_POLICY.TYPE:
            lr = 1
        else:
            lr = cfg.SOLVER.BASE_LR

        # params = model.parameters()
        params_1 = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
        params_2 = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
        params = [{'params': params_1, 'lr': 0.1}, {'params': params_2, 'lr': 1}]
        print(len(params_1), len(params_2))

        # 优化器设置
        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, 
                lr = lr, 
                momentum = cfg.SOLVER.SGD.MOMENTUM,
                nesterov = True
            )
        elif cfg.SOLVER.TYPE == 'ADAM':
            # 初始lr在scheduler为Noam时无效
            self.optimizer = torch.optim.Adam(
                params,
                lr = lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(
                params,
                lr = lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = lr
            )
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(
                params, 
                lr = lr
            )
        elif cfg.SOLVER.TYPE == 'RADAM':
            self.optimizer = RAdam(
                params, 
                lr = lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        else:
            raise NotImplementedError

        # 学习率策略设置
        if cfg.SOLVER.LR_POLICY.TYPE == 'Fix':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size = cfg.SOLVER.LR_POLICY.STEP_SIZE, 
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,  
                factor = cfg.SOLVER.LR_POLICY.PLATEAU_FACTOR, 
                patience = cfg.SOLVER.LR_POLICY.PLATEAU_PATIENCE
            ) 
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Lambda_xe': # 
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,  
                lambda_lr_xe
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Lambda_rl': # 
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,  
                lambda_lr_rl
            )           
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Noam':
            self.scheduler = lr_scheduler.create(
                'Noam', 
                self.optimizer,
                model_size = cfg.SOLVER.LR_POLICY.MODEL_SIZE,
                factor = cfg.SOLVER.LR_POLICY.FACTOR,
                warmup = cfg.SOLVER.LR_POLICY.WARMUP,
                last_epoch = self.last_epoch
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'MultiStep':
            self.scheduler = lr_scheduler.create(
                'MultiStep', 
                self.optimizer,
                milestones = cfg.SOLVER.LR_POLICY.STEPS,
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, lrs_type, val=None):
        if self.scheduler is None:
            return

        if cfg.SOLVER.LR_POLICY.TYPE != 'Plateau':
            val = None

        if lrs_type == cfg.SOLVER.LR_POLICY.SETP_TYPE:
            self.scheduler.step(val)

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        # lr = sorted(list(set(lr)))
        return lr

    
def lambda_lr_xe(s):
    xe_base_lr = cfg.SOLVER.BASE_LR

    if s <= 3:
        lr = xe_base_lr * s / 4 # 0.0001 0.0002 0.0003 0.0004
    elif s <= 9:
        lr = xe_base_lr # 0.0004
    elif s <= 13:
        lr = xe_base_lr * 0.2 # 8e-5
    else:
        lr = xe_base_lr * 0.2 * 0.2 # 1.6e-5

    return lr
    
def lambda_lr_rl(s):
    rl_base_lr = cfg.SOLVER.BASE_LR

    if s <= 7:
        lr = rl_base_lr # 2e-5 # 16-22
    elif s <= 13:
        lr = rl_base_lr * 0.2 # 4e-6 # 22-27
    elif s <= 18:
        lr = rl_base_lr * 0.2 * 0.2 # 8e-7 # 27-32
    else:
        lr = rl_base_lr * 0.2 * 0.2 * 0.2 # 1.6e-7

    return lr
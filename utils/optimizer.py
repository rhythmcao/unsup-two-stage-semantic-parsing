#coding=utf8
'''
    Set optimizer for train_model
'''
import torch
import torch.nn as nn
from torch.optim import Adam, Adadelta

def set_optimizer(*args, lr=1e-3, l2=1e-5, max_norm=5, **kargs):
    params = []
    for train_model in args:
        params += list(train_model.named_parameters())
    grouped_params = [
        {'params': list(set([p for n, p in params if p.requires_grad and 'bias' not in n])), 'weight_decay': l2},
        {'params': list(set([p for n, p in params if p.requires_grad and 'bias' in n])), 'weight_decay': 0.0}
    ]
    optimizer = MyAdam(grouped_params, lr=lr, max_norm=max_norm)
    return optimizer

class MyAdam(Adam):
    """
        Add clip_grad_norm_ for Optimizer Adam
    """
    def __init__(self, *args, **kargs):
        self.max_norm = kargs.pop('max_norm', -1)
        super(MyAdam, self).__init__(*args, **kargs)

    def step(self, *args, **kargs):
        if self.max_norm > 0:
            for group in self.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.max_norm)
        super(MyAdam, self).step(*args, **kargs)

def set_optimizer_adadelta(*args, lr=1., l2=1e-5, max_norm=5):
    params = []
    for train_model in args:
        params += list(train_model.named_parameters())
    grouped_params = [
        {'params': list(set([p for n, p in params if p.requires_grad and 'bias' not in n])), 'weight_decay': l2},
        {'params': list(set([p for n, p in params if p.requires_grad and 'bias' in n])), 'weight_decay': 0.0}
    ]
    optimizer = MyAdadelta(grouped_params, lr=lr, max_norm=max_norm)
    return optimizer

class MyAdadelta(Adadelta):
    """
        Add clip_grad_norm_ for Optimizer Adadelta
    """
    def __init__(self, *args, **kargs):
        self.max_norm = kargs.pop('max_norm', -1)
        super(MyAdadelta, self).__init__(*args, **kargs)

    def step(self, *args, **kargs):
        if self.max_norm > 0:
            for group in self.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.max_norm)
        super(MyAdadelta, self).step(*args, **kargs)

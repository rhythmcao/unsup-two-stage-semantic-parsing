#coding=utf8
'''
    Set loss function, add label smoothing
'''
import torch
import torch.nn as nn

def set_loss_function(smoothing=0.0, reduction='sum', ignore_index=-100):
    loss_function = LabelSmoothing(smoothing=smoothing, reduction=reduction, ignore_index=ignore_index)
    return loss_function

class LabelSmoothing(nn.Module):
    """
        Implement label smoothing CE loss.
    """
    def __init__(self, smoothing=0.0, ignore_index=-100, reduction='sum'):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        assert x.dim() == 3 and target.dim() == 2
        size = x.size(-1)
        true_dist = x.new_zeros(x.size())
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        true_dist += self.smoothing / (size - 1) # minus 1 due to ignore_index
        true_dist[:, :, self.ignore_index] = 0
        true_dist.masked_fill_(target.unsqueeze(-1) == self.ignore_index, 0. )
        true_dist = true_dist.detach()
        return self.criterion(x, true_dist)

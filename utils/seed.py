#coding=utf8
import random, torch
import numpy as np

def set_random_seed(random_seed=999):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
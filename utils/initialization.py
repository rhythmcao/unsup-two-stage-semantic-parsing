#coding=utf8
"""
    Utility functions include:
        1. set random seed for all libs
        2. set output logger path
        3. select torch.device
"""
import sys, logging, random, torch
import numpy as np
from utils.hyperparam import hyperparam_path


def set_logger(exp_path, testing=False, append_mode=False):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    level = logging.DEBUG
    logger.setLevel(level)
    mode = 'a' if append_mode else 'w'
    fileHandler = logging.FileHandler('%s/log_%s.txt' % (exp_path, ('test' if testing else 'train')), mode=mode)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger


def set_random_seed(random_seed=999):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


def set_torch_device(deviceId):
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        torch.backends.cudnn.enabled = False
        # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # used when debug
        ## These two lines are used to ensure reproducibility with cudnn backend
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return device


def initialization_wrapper(args):
    set_random_seed(args.seed)
    exp_path = hyperparam_path(args)
    logger = set_logger(exp_path, args.testing)
    device = set_torch_device(args.deviceId)
    logger.info(f"Initialization finished ...")
    logger.info(f"Random seed is set to {args.seed}")
    logger.info(f"Output path is {exp_path}")
    return exp_path, logger, device
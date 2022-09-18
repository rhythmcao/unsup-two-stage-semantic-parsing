# coding=utf8
import argparse, os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from utils.vocab import Vocab
from utils.seed import set_random_seed
from utils.logger import set_logger
from utils.optimizer import set_optimizer_adadelta
from utils.gpu import set_torch_device
from utils.constants import *
from utils.solver.solver_discriminator import CLSFSolver
from utils.word2vec import load_embeddings
from utils.example import Example, UtteranceExample, split_dataset
from utils.hyperparam import hyperparam_classifier
from models.classifier import StyleClassifier as model

############################### Arguments parsing and Preparations ##############################

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='discriminator', help='classification model')
    parser.add_argument('--dataset', required=True, help='which dataset to experiment on')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    parser.add_argument('--read_model_path', help='Testing mode, load style classifier model path')
    # model paras
    parser.add_argument('--emb_size', type=int, default=100, help='embedding size')
    parser.add_argument('--filters', type=int, nargs='+', default=[3, 4, 5], help='filter size')
    parser.add_argument('--filters_num', type=int, nargs='+', default=[10, 20, 30], help='filter num')
    # training paras
    parser.add_argument('--lr', type=float, default=1, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay rate')
    parser.add_argument('--reduction', choices=['mean', 'sum'], default='sum')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate at each non-recurrent layer')
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--test_batchSize', type=int, default=128, help='input batch size in decoding')
    parser.add_argument('--init_weight', type=float, default=0.2, help='all weights will be set to [-init_weight, init_weight] during initialization')
    parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
    # special paras
    parser.add_argument('--labeled', type=float, default=1.0, help='training use only this propotion of dataset')
    parser.add_argument('--deviceId', type=int, default=0, help='train model on ith gpu. -1:cpu')
    parser.add_argument('--seed', type=int, default=999, help='set initial random seed')
    opt = parser.parse_args(args)
    if opt.testing:
        assert opt.read_model_path
    return opt

opt = main()

####################### Output path, logger, device and random seed configuration #################

exp_path = opt.read_model_path if opt.testing else hyperparam_classifier(opt)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logger = set_logger(exp_path, testing=opt.testing)
logger.info("Parameters: " + str(json.dumps(vars(opt), indent=4)))
logger.info("Experiment path: %s" % (exp_path))
opt.device = set_torch_device(opt.deviceId)
set_random_seed(opt.seed)

################################ Vocab and Data Reader ###########################

clsf_vocab = Vocab(opt.dataset, task='discriminator')
logger.info("Vocab size for style classifier is: %s" % (len(clsf_vocab.nl2id)))

logger.info("Read dataset starts at %s" % (time.asctime(time.localtime(time.time()))))
Example.set_domain(opt.dataset)
if not opt.testing:
    train_dataset, dev_dataset = Example.load_dataset(choice='train')
    train_dataset, _ = split_dataset(train_dataset, opt.labeled)
    train_dataset, dev_dataset = UtteranceExample.from_dataset(train_dataset), UtteranceExample.from_dataset(dev_dataset)
    logger.info("Train and dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))
test_dataset = Example.load_dataset(choice='test')
test_dataset = UtteranceExample.from_dataset(test_dataset)
logger.info("Test dataset size is: %s" % (len(test_dataset)))

###################################### Model Construction ########################################

if not opt.testing:
    clsf_params = {
        'emb_size': opt.emb_size, 'vocab_size': len(clsf_vocab.nl2id), 'pad_token_idxs': [clsf_vocab.nl2id[PAD]],
        'filters': opt.filters, 'filters_num': opt.filters_num, 'dropout' : opt.dropout, "init": opt.init_weight
    }
    json.dump(clsf_params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else:
    clsf_params = json.load(open(os.path.join(exp_path, 'params.json'), 'r'))
clsf_model = model(**clsf_params)
clsf_model = clsf_model.to(opt.device)

##################################### Model Initialization #########################################

if not opt.testing:
    ratio = load_embeddings(clsf_model.embedding, clsf_vocab.nl2id, opt.device)
    logger.info("%.2f%% word embeddings from pretrained vectors" % (ratio * 100))
else:
    model_path = os.path.join(opt.read_model_path, 'model.pkl')
    clsf_model.load_model(model_path)
    logger.info("Load CF Classifier from path %s" % (model_path))

# set loss function and optimizer
loss_function = nn.BCELoss(reduction=opt.reduction)
optimizer = set_optimizer_adadelta(clsf_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm)

###################################### Training and Decoding #######################################

solver = CLSFSolver(clsf_model, clsf_vocab, loss_function, optimizer, exp_path, logger, device=opt.device)
if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    solver.train_and_decode(train_dataset, dev_dataset, test_dataset,
        batchSize=opt.batchSize, test_batchSize=opt.test_batchSize, max_epoch=opt.max_epoch)
else:
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    acc = solver.decode(test_dataset, os.path.join(exp_path, 'test.eval'), opt.test_batchSize)
    logger.info('Evaluation cost: %.4fs\tNL/CF acc : %.4f' % (time.time() - start_time, acc))

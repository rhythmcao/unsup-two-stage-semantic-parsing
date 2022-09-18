#coding=utf8
import argparse, os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hyperparam import hyperparam_seq2seq
from utils.logger import set_logger
from utils.vocab import Vocab
from utils.seed import set_random_seed
from utils.example import split_dataset, Example
from utils.constants import PAD, UNK
from utils.loss import set_loss_function
from utils.optimizer import set_optimizer
from utils.gpu import set_torch_device
from models.constructor import construct_model as model
from utils.word2vec import load_embeddings
from utils.solver.solver_pipeline_semantic_parsing import PSPSolver

############################### Arguments parsing and Preparations ##############################

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='pipeline semantic parsing')
    parser.add_argument('--dataset', type=str, required=True, help='which dataset to experiment on')
    parser.add_argument('--read_nsp_model_path', required=True, help='pretrained naive semantic parser model path')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    # pretrained models
    parser.add_argument('--read_model_path', required=False, help='Read model and hyperparams from this path')
    # model paras
    parser.add_argument('--emb_size', type=int, default=100, help='embedding size')
    parser.add_argument('--hidden_dim', type=int, default=200, help='hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--cell', default='lstm', choices=['lstm', 'gru'], help='rnn cell choice')
    parser.add_argument('--trans', default='empty', choices=['empty', 'tanh(affine)'])
    # training paras
    parser.add_argument('--reduction', default='sum', choices=['mean', 'sum'], help='loss function argument')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate at each non-recurrent layer')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--test_batchSize', type=int, default=128, help='input batch size in decoding')
    parser.add_argument('--init_weight', type=float, default=0.2, help='all weights will be set to [-init_weight, init_weight] during initialization')
    parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
    parser.add_argument('--beam', default=5, type=int, help='beam search size')
    parser.add_argument('--n_best', default=1, type=int, help='return n best results')
    # special paras
    parser.add_argument('--labeled', type=float, default=1.0, help='training use only this propotion of dataset')
    parser.add_argument('--deviceId', type=int, default=0, help='train model on ith gpu. -1: cpu, o.w. gpu index')
    parser.add_argument('--seed', type=int, default=999, help='set initial random seed')
    opt = parser.parse_args(args)
    if opt.testing:
        assert opt.read_model_path
    return opt

opt = main()

####################### Output path, logger, device and random seed configuration #################

exp_path = opt.read_model_path if opt.testing else hyperparam_seq2seq(opt)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logger = set_logger(exp_path, testing=opt.testing)
logger.info("Parameters: " + str(json.dumps(vars(opt), indent=4)))
logger.info("Experiment path: %s" % (exp_path))
opt.device = set_torch_device(opt.deviceId)
set_random_seed(opt.seed)

################################ Vocab and Data Reader ###########################
nl2cf_vocab, sp_vocab = Vocab(opt.dataset, task='nl2cf'), Vocab(opt.dataset, task='semantic_parsing')
logger.info("Vocab size for input natural language sentence is: %s" % (len(nl2cf_vocab.nl2id)))
logger.info("Vocab size for intermediate canonical forms is: %s" % (len(nl2cf_vocab.cf2id)))
logger.info("Vocab size for output logical form is: %s" % (len(sp_vocab.lf2id)))

logger.info("Read dataset %s starts at %s" % (opt.dataset, time.asctime(time.localtime(time.time()))))
Example.set_domain(opt.dataset)
if not opt.testing:
    train_dataset, dev_dataset = Example.load_dataset(choice='train')
    train_dataset, _ = split_dataset(train_dataset, opt.labeled)
    logger.info("Train and dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))
test_dataset = Example.load_dataset(choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

###################################### Model Construction ########################################

if not opt.testing:
    params = {
        "src_vocab": len(nl2cf_vocab.nl2id), "tgt_vocab": len(nl2cf_vocab.cf2id),
        "pad_src_idxs": [nl2cf_vocab.nl2id[PAD]], "pad_tgt_idxs": [nl2cf_vocab.cf2id[PAD]],
        "src_emb_size": opt.emb_size, "tgt_emb_size": opt.emb_size, "hidden_dim": opt.hidden_dim, "trans": opt.trans,
        "num_layers": opt.num_layers, "cell": opt.cell, "dropout": opt.dropout, "init": opt.init_weight,
        "read_nsp_model_path": opt.read_nsp_model_path
    }
    json.dump(params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else:
    params = json.load(open(os.path.join(exp_path, 'params.json'), 'r'))

train_model = model(**params)
train_model = train_model.to(opt.device)
sp_params = json.load(open(os.path.join(params["read_nsp_model_path"], 'params.json'), 'r'))
sp_model = model(**sp_params)
sp_model_path = os.path.join(params["read_nsp_model_path"], 'model.pkl')
sp_model.load_model(sp_model_path)
logger.info("Load Naive Semantic Parser from path %s" % (sp_model_path))
sp_model = sp_model.to(opt.device)

##################################### Model Initialization #########################################

if not opt.testing:
    ratio = load_embeddings(train_model.src_embed.embed, nl2cf_vocab.nl2id, opt.device)
    logger.info("%.2f%% input natural language word embeddings from pretrained vectors" % (ratio * 100))
    ratio = load_embeddings(train_model.tgt_embed.embed, nl2cf_vocab.cf2id, opt.device)
    logger.info("%.2f%% intermediate canonical forms word embeddings from pretrained vectors" % (ratio * 100))
else:
    model_path = os.path.join(opt.read_model_path, 'model.pkl')
    train_model.load_model(model_path)
    logger.info("Load model from path %s" % (model_path))

# set loss function and optimizer
loss_function = set_loss_function(ignore_index=nl2cf_vocab.cf2id[PAD], reduction=opt.reduction)
optimizer = set_optimizer(train_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm)

###################################### Training and Decoding #######################################

solver = PSPSolver(train_model, sp_model, nl2cf_vocab, sp_vocab, loss_function, optimizer, exp_path, logger, device=opt.device)
if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    solver.train_and_decode(train_dataset, dev_dataset, test_dataset,
        batchSize=opt.batchSize, test_batchSize=opt.test_batchSize,
        max_epoch=opt.max_epoch, beam=opt.beam, n_best=opt.n_best)
else:
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    cf_acc, lf_acc = solver.decode(test_dataset, os.path.join(exp_path, 'test.eval'),
        opt.test_batchSize, beam=opt.beam, n_best=opt.n_best)
    logger.info('Evaluation cost: %.4fs\tCF/LF Acc : %.4f/%.4f' % (time.time() - start_time, cf_acc, lf_acc))
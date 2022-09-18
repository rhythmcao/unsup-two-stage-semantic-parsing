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
from utils.solver.solver_faked_sample_two_stage import FSSPSolver

############################### Arguments parsing and Preparations ##############################

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='faked sample two stage semantic parsing')
    parser.add_argument('--dataset', type=str, required=True, help='which dataset to experiment on')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    # pretrained models
    parser.add_argument('--read_model_path', required=False, help='Read model and hyperparams from this path')
    parser.add_argument('--read_nsp_model_path', required=True, help='path to naive semantic parsing model')
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
    parser.add_argument('--shared_encoder', action='store_true', help='whether share the encoder for NL2CF and CF2NL models')
    parser.add_argument('--method', choices=['wmd', 'bow'], default='wmd', help='how to create faked samples')
    parser.add_argument('--labeled', type=float, default=0., help='training use only this propotion of dataset')
    parser.add_argument('--deviceId', type=int, help='train model on ith gpu. -1: cpu, o.w. gpu index')
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
nl2cf_device = cf2nl_device = sp_device = set_torch_device(opt.deviceId)
set_random_seed(opt.seed)

################################ Vocab and Data Reader ###########################
nl2cf_vocab, cf2nl_vocab, sp_vocab = Vocab(opt.dataset, task="nl2cf"), Vocab(opt.dataset, task='cf2nl'), Vocab(opt.dataset, task='semantic_parsing')
logger.info("NL2CF: Vocab size for input natural language sentence is: %s" % (len(nl2cf_vocab.nl2id)))
logger.info("NL2CF: Vocab size for output canonical form is: %s" % (len(nl2cf_vocab.cf2id)))
logger.info("CF2NL: Vocab size for input canonical form is: %s" % (len(cf2nl_vocab.cf2id)))
logger.info("CF2NL: Vocab size for output natural language sentence is: %s" % (len(cf2nl_vocab.nl2id)))

logger.info("Read dataset %s starts at %s" % (opt.dataset, time.asctime(time.localtime(time.time()))))
Example.set_domain(opt.dataset, add=(opt.method == 'wmd'))
if not opt.testing:
    train_dataset, dev_dataset = Example.load_dataset(choice='train')
    unlabeled_nl, _ = split_dataset(train_dataset, opt.labeled)
    unlabeled_cf = unlabeled_nl
    logger.info("Train and dev dataset size is: %s and %s" % (len(unlabeled_nl), len(dev_dataset)))
test_dataset = Example.load_dataset(choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

###################################### Model Construction ########################################

if not opt.testing:
    nl2cf_params = {
        "src_vocab": len(nl2cf_vocab.nl2id), "tgt_vocab": len(nl2cf_vocab.cf2id),
        "pad_src_idxs": [nl2cf_vocab.nl2id[PAD]], "pad_tgt_idxs": [nl2cf_vocab.cf2id[PAD]],
        "src_emb_size": opt.emb_size, "tgt_emb_size": opt.emb_size, "hidden_dim": opt.hidden_dim, "trans": opt.trans,
        "num_layers": opt.num_layers, "cell": opt.cell, "dropout": opt.dropout, "init": opt.init_weight,
        "shared_encoder": opt.shared_encoder, "method": opt.method
    }
    json.dump(nl2cf_params, open(os.path.join(exp_path, 'nl2cf_params.json'), 'w'), indent=4)
    cf2nl_params = {
        "src_vocab": len(cf2nl_vocab.cf2id), "tgt_vocab": len(cf2nl_vocab.nl2id),
        "pad_src_idxs": [cf2nl_vocab.cf2id[PAD]], "pad_tgt_idxs": [cf2nl_vocab.nl2id[PAD]],
        "src_emb_size": opt.emb_size, "tgt_emb_size": opt.emb_size, "hidden_dim": opt.hidden_dim, "trans": opt.trans,
        "num_layers": opt.num_layers, "cell": opt.cell, "dropout": opt.dropout, "init": opt.init_weight
    }
    json.dump(cf2nl_params, open(os.path.join(exp_path, 'cf2nl_params.json'), 'w'), indent=4)
else:
    nl2cf_params = json.load(open(os.path.join(exp_path, 'nl2cf_params.json'), 'r'))
    cf2nl_params = json.load(open(os.path.join(exp_path, 'cf2nl_params.json'), 'r'))

nl2cf_model = model(**nl2cf_params)
cf2nl_model = model(**cf2nl_params)
sp_params = json.load(open(os.path.join(opt.read_nsp_model_path, 'params.json'), 'r'))
sp_model = model(**sp_params)
nl2cf_model = nl2cf_model.to(nl2cf_device)
cf2nl_model = cf2nl_model.to(cf2nl_device)
sp_model = sp_model.to(sp_device)

##################################### Model Initialization #########################################

if nl2cf_params["shared_encoder"]:
    cf2nl_model.src_embed = nl2cf_model.src_embed
    cf2nl_model.encoder = nl2cf_model.encoder

model_path = os.path.join(opt.read_nsp_model_path, 'model.pkl')
sp_model.load_model(model_path)
logger.info("Load Naive Semantic Parsing model from path %s" % (model_path))

if not opt.testing:
    ratio = load_embeddings(nl2cf_model.src_embed.embed, nl2cf_vocab.nl2id, nl2cf_device)
    logger.info("NL2CF %.2f%% input word embeddings from pretrained vectors" % (ratio * 100))
    ratio = load_embeddings(nl2cf_model.tgt_embed.embed, nl2cf_vocab.cf2id, nl2cf_device)
    logger.info("NL2CF %.2f%% output canonical form word embeddings from pretrained vectors" % (ratio * 100))
    if not nl2cf_params["shared_encoder"]:
        ratio = load_embeddings(cf2nl_model.src_embed.embed, cf2nl_vocab.cf2id, cf2nl_device)
        logger.info("CF2NL %.2f%% input canonical form word embeddings from pretrained vectors" % (ratio * 100))
    ratio = load_embeddings(cf2nl_model.tgt_embed.embed, cf2nl_vocab.nl2id, cf2nl_device)
    logger.info("CF2NL %.2f%% output word embeddings from pretrained vectors" % (ratio * 100))
else:
    model_path = os.path.join(opt.read_model_path, 'nl2cf_model.pkl')
    nl2cf_model.load_model(model_path)
    logger.info("Load NL2CF model from path %s" % (model_path))
    model_path = os.path.join(opt.read_model_path, 'cf2nl_model.pkl')
    cf2nl_model.load_model(model_path)
    logger.info("Load CF2NL model from path %s" % (model_path))

# set loss function and optimizer
loss_function = {"nl2cf": None, "cf2nl": None}
loss_function["nl2cf"] = set_loss_function(ignore_index=nl2cf_vocab.cf2id[PAD], reduction=opt.reduction)
loss_function["cf2nl"] = set_loss_function(ignore_index=cf2nl_vocab.nl2id[PAD], reduction=opt.reduction)
optimizer = set_optimizer(nl2cf_model, cf2nl_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm)

###################################### Training and Decoding #######################################

vocab = {'nl2cf': nl2cf_vocab, 'cf2nl': cf2nl_vocab, 'sp': sp_vocab}
device = {'nl2cf': nl2cf_device, 'cf2nl': cf2nl_device, 'sp': sp_device}
solver = FSSPSolver(nl2cf_model, cf2nl_model, sp_model, vocab, loss_function, optimizer, exp_path, logger, device, method=nl2cf_params["method"])
if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    solver.train_and_decode(unlabeled_nl, unlabeled_cf, dev_dataset, test_dataset,
        batchSize=opt.batchSize, test_batchSize=opt.test_batchSize,
        max_epoch=opt.max_epoch, beam=opt.beam, n_best=opt.n_best)
else:
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    cf_acc, lf_acc, nl_bleu = solver.decode(test_dataset, os.path.join(exp_path, 'test.eval'),
        opt.test_batchSize, beam=opt.beam, n_best=opt.n_best)
    logger.info('Evaluation cost: %.4fs\tCF/LF Acc : %.4f/%.4f\tNL Bleu : %.4f' % (time.time() - start_time, cf_acc, lf_acc, nl_bleu))

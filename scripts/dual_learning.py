#coding=utf8
import argparse, os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vocab import Vocab
from utils.example import Example, split_dataset
from utils.optimizer import set_optimizer
from utils.loss import set_loss_function
from utils.seed import set_random_seed
from utils.logger import set_logger
from utils.gpu import set_torch_device
from utils.constants import *
from utils.solver.solver_dual_learning import DualLearningSolver
from utils.hyperparam import hyperparam_dual_learning
from models.constructor import construct_model as model
from models.classifier import StyleClassifier as Classifier
from models.dual_learning import DualLearning
from models.reward import RewardModel
from models.language_model import LanguageModel

############################### Arguments parsing and Preparations ##############################

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='dual paraphrase model for unsupervised semantic parsing')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    parser.add_argument('--dataset', required=True, help='which dataset to experiment on')
    parser.add_argument('--read_model_path', help='Testing mode, load nl2cf and cf2nl model path')
    # model params
    parser.add_argument('--read_pretrained_model_path', help='pretrained nl2cf and cf2nl model')
    parser.add_argument('--read_nsp_model_path', required=True, help='load naive semantic parsing model')
    parser.add_argument('--read_lm_path', help='language model')
    parser.add_argument('--read_sty_path', help='style classfier')
    # pseudo training paras
    parser.add_argument('--reduction', choices=['sum', 'mean'], default='sum')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--test_batchSize', type=int, default=128, help='input batch size in decoding')
    parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
    # special paras
    parser.add_argument('--sample', type=int, default=6, help='size of sampling during training in dual learning')
    parser.add_argument('--beam', default=5, type=int, help='used during decoding time')
    parser.add_argument('--n_best', default=1, type=int, help='used during decoding time')
    parser.add_argument('--alpha', type=float, default=0.5, help='coefficient which combines nl2cf valid and reconstruction reward')
    parser.add_argument('--beta', type=float, default=0.5, help='coefficient which combines cf2nl valid and reconstruction reward')
    parser.add_argument('--reward', choices=['flu', 'sty', 'rel', 'flu+sty', 'flu+rel', 'sty+rel', 'flu+sty+rel'], default='flu+sty+rel', help='reward choice during drl')
    parser.add_argument('--scheme', type=str, default='bt+drl', help='schemes used during cycle learning phase')
    parser.add_argument('--labeled', type=float, default=1.0, help='ratio of labeled samples')
    parser.add_argument('--nl_labeled', type=float, default=1.0, help='ratio of labeled natural language utterances we used')
    parser.add_argument('--deviceId', type=int, nargs=2, default=[-1, -1], help='device for nl2cf and cf2nl model respectively if not shared_encoder')
    parser.add_argument('--seed', type=int, default=999, help='set initial random seed')
    opt = parser.parse_args(args)
    return opt

opt = main()

####################### Output path, logger, device and random seed configuration #################

if not opt.testing:
    nl2cf_params = json.load(open(os.path.join(opt.read_pretrained_model_path, 'nl2cf_params.json'), 'r'))
    params = {
        "read_pretrained_model_path": opt.read_pretrained_model_path, "read_sty_path": opt.read_sty_path,
        "read_lm_path": opt.read_lm_path, "shared_encoder": nl2cf_params["shared_encoder"], "noisy_channel": nl2cf_params["noisy_channel"],
        "sample": opt.sample, "alpha": opt.alpha, "beta": opt.beta, "reduction": opt.reduction, "scheme": opt.scheme, "reward": opt.reward
    }
else:
    params = json.load(open(os.path.join(opt.read_model_path, "params.json"), 'r'))

opt.shared_encoder, opt.noisy_channel = params['shared_encoder'], params['noisy_channel']
exp_path = opt.read_model_path if opt.testing else hyperparam_dual_learning(opt)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
json.dump(params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)

logger = set_logger(exp_path, testing=opt.testing)
logger.info("Parameters: " + str(json.dumps(vars(opt), indent=4)))
logger.info("Experiment path: %s" % (exp_path))
if opt.shared_encoder: # try to allocate equal gpu memory
    cf2nl_device = nl2cf_device = set_torch_device(opt.deviceId[0])
    sty_device = sp_device = lm_device = set_torch_device(opt.deviceId[1])
else:
    sp_device = nl2cf_device = set_torch_device(opt.deviceId[0])
    sty_device = cf2nl_device = lm_device = set_torch_device(opt.deviceId[1])
set_random_seed(opt.seed)

################################ Vocab and Data Reader ###########################

nl2cf_vocab, cf2nl_vocab = Vocab(opt.dataset, task='nl2cf'), Vocab(opt.dataset, task='cf2nl')
lm_vocab, sp_vocab, sty_vocab = Vocab(opt.dataset, task='language_model'), Vocab(opt.dataset, task='semantic_parsing'), Vocab(opt.dataset, task='discriminator')
logger.info("NL2CF model vocabulary ...")
logger.info("Vocab size for input natural language sentence is: %s" % (len(nl2cf_vocab.nl2id)))
logger.info("Vocab size for output canonical form is: %s" % (len(nl2cf_vocab.cf2id)))

logger.info("CF2NL model vocabulary ...")
logger.info("Vocab size for input canonical form is: %s" % (len(cf2nl_vocab.cf2id)))
logger.info("Vocab size for output natural language sentence is: %s" % (len(cf2nl_vocab.nl2id)))

logger.info("Read dataset starts at %s" % (time.asctime(time.localtime(time.time()))))
drop, add, shuffle = ('drop' in params['noisy_channel'] and 'dae' in params['scheme']), ('add' in params['noisy_channel'] and 'dae' in params['scheme']), ('shuffle' in params['noisy_channel'] and 'dae' in params['scheme'])
Example.set_domain(opt.dataset, drop=drop, add=add, shuffle=shuffle)
if not opt.testing:
    train_dataset, dev_dataset = Example.load_dataset('train')
    unlabeled_nl = unlabeled_cf = train_dataset
    unlabeled_nl, _ = split_dataset(unlabeled_nl, opt.nl_labeled)
    labeled_dataset, _ = split_dataset(train_dataset, opt.labeled)
    logger.info("Train/Dev dataset size is: %s/%s" % (len(train_dataset), len(dev_dataset)))
    logger.info("Labeled dataset size is: %s" % (len(labeled_dataset)))
test_dataset = Example.load_dataset('test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

###################################### Model Construction ########################################

nl2cf_params = json.load(open(os.path.join(params["read_pretrained_model_path"], 'nl2cf_params.json'), 'r'))
cf2nl_params = json.load(open(os.path.join(params["read_pretrained_model_path"], 'cf2nl_params.json'), 'r'))
sp_params = json.load(open(os.path.join(opt.read_nsp_model_path, 'params.json'), 'r'))
sp_model_path = os.path.join(opt.read_nsp_model_path, 'model.pkl')
nl_params = json.load(open(os.path.join(params["read_lm_path"], 'nl_params.json'), 'r'))
cf_params = json.load(open(os.path.join(params["read_lm_path"], 'cf_params.json'), 'r'))
nl_model_path = os.path.join(params['read_lm_path'], 'nl_model.pkl')
cf_model_path = os.path.join(params['read_lm_path'], 'cf_model.pkl')
sty_params = json.load(open(os.path.join(params['read_sty_path'], 'params.json'), 'r'))
sty_model_path = os.path.join(params['read_sty_path'], 'model.pkl')

nl2cf_model, cf2nl_model = model(**nl2cf_params), model(**cf2nl_params)
nl_model, cf_model = LanguageModel(**nl_params), LanguageModel(**cf_params)
nl_model.load_model(nl_model_path)
logger.info("Load Natural Language Model from path %s" % (nl_model_path))
cf_model.load_model(cf_model_path)
logger.info("Load Canonical Form Language Model from path %s" % (cf_model_path))
sp_model = model(**sp_params)
sp_model.load_model(sp_model_path)
logger.info("Load Semantic Parsing Model from path %s" % (sp_model_path))
sty_model = Classifier(**sty_params)
sty_model.load_model(sty_model_path)
logger.info("Load Style Classification Model from path %s" % (sty_model_path))
if not opt.testing:
    nl2cf_model_path = os.path.join(params["read_pretrained_model_path"], 'nl2cf_model.pkl')
    nl2cf_model.load_model(nl2cf_model_path)
    logger.info("Load NL2CF model from path %s" % (nl2cf_model_path))
    cf2nl_model_path = os.path.join(params["read_pretrained_model_path"], 'cf2nl_model.pkl')
    cf2nl_model.load_model(cf2nl_model_path)
    logger.info("Load CF2NL model from path %s" % (cf2nl_model_path))
else:
    nl2cf_model.load_model(os.path.join(exp_path, 'nl2cf_model.pkl'))
    logger.info("Load NL2CF model from path %s" % (exp_path))
    cf2nl_model.load_model(os.path.join(exp_path, 'cf2nl_model.pkl'))
    logger.info("Load CF2NL model from path %s" % (exp_path))
reward_model = RewardModel(nl_model, cf_model, sp_model, sty_model, lm_vocab, sp_vocab, sty_vocab,
    lm_device=lm_device, sp_device=sp_device, sty_device=sty_device, reward=params["reward"])
train_model = DualLearning(nl2cf_model, cf2nl_model, reward_model, nl2cf_vocab, cf2nl_vocab,
    alpha=params['alpha'], beta=params['beta'], sample=params['sample'], shared_encoder=params["shared_encoder"],
    reduction=params["reduction"], nl2cf_device=nl2cf_device, cf2nl_device=cf2nl_device)

loss_function = {'nl2cf': {}, 'cf2nl': {}}
loss_function['nl2cf'] = set_loss_function(ignore_index=nl2cf_vocab.cf2id[PAD], reduction=opt.reduction)
loss_function['cf2nl'] = set_loss_function(ignore_index=cf2nl_vocab.nl2id[PAD], reduction=opt.reduction)
optimizer = set_optimizer(nl2cf_model, cf2nl_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm)

###################################### Training and Decoding #######################################

vocab = {'nl2cf': nl2cf_vocab, 'cf2nl': cf2nl_vocab, 'sp': sp_vocab}
device = {'nl2cf': nl2cf_device, 'cf2nl': cf2nl_device, 'sp': sp_device}
solver = DualLearningSolver(train_model, sp_model, vocab, loss_function, optimizer, exp_path, logger, device=device)
if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    solver.train_and_decode(unlabeled_nl, unlabeled_cf, labeled_dataset, dev_dataset, test_dataset,
        batchSize=opt.batchSize, test_batchSize=opt.test_batchSize, scheme=params["scheme"],
        max_epoch=opt.max_epoch, beam=opt.beam, n_best=opt.n_best)
else:
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    cf_acc, lf_acc, nl_bleu = solver.decode(test_dataset, os.path.join(exp_path, 'test.eval'), opt.test_batchSize, beam=opt.beam, n_best=opt.n_best)
    logger.info('Evaluation cost: %.4fs\tNL2CF (CF/LF Acc : %.4f)\tCF2NL (Bleu: %.4f)'
        % (time.time() - start_time, cf_acc, lf_acc, nl_bleu))

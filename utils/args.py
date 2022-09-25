#coding=utf8
import argparse, sys
from utils.constants import DOMAINS

def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    args = arg_parser.parse_args(params)
    return args


def add_argument_base(parser):
    parser.add_argument('--task', type=str, help='semantic parsing task')
    parser.add_argument('--dataset', type=str, required=True, choices=DOMAINS, help='which dataset to experiment on')
    parser.add_argument('--read_model_path', help='Read model and hyperparams from this path')
    parser.add_argument('--read_pdp_model_path', help='Read pretrained dual paraphrase model and hyperparams from this path')
    parser.add_argument('--read_nsp_model_path', help='Read naive semantic parsing model and hyperparams from this path')
    parser.add_argument('--read_language_model_path', help='Read language model and hyperparams from this path')
    parser.add_argument('--read_tsc_model_path', help='Read text style classifier and hyperparams from this path')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    # model params
    parser.add_argument('--pretrained_embed', type=str, default='glove', choices=['glove', 'elmo', 'bert'], help='pretrained embedding type')
    parser.add_argument('--embed_size', type=int, default=100, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--cell', default='lstm', choices=['lstm', 'gru'], help='rnn cell choice')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate at each non-recurrent layer')
    parser.add_argument('--filters', type=int, nargs='+', default=[3, 4, 5], help='filter size for text CNN model')
    parser.add_argument('--filters_num', type=int, nargs='+', default=[10, 20, 30], help='filter num for text CNN model')
    parser.add_argument('--share_encoder', action='store_true', help='whether share the encoder for natural language and canonical form')
    parser.add_argument('--init_weight', type=float, help='all weights will be set to [-init_weight, init_weight] during initialization')
    # training params
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--layerwise_decay', type=float, default=1., help='learning rate layerwise decay rate for pre-trained language models')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    parser.add_argument('--lr_schedule', default='constant', choices=['constant', 'constant_warmup', 'linear'], help='lr scheduler')
    parser.add_argument('--eval_after_epoch', default=60, type=int, help='Start to evaluate after x 1000 iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='input batch size in decoding')
    parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
    parser.add_argument('--beam_size', default=5, type=int, help='beam search size')
    parser.add_argument('--n_best', default=1, type=int, help='return n best results')
    # special params
    parser.add_argument('--train_scheme', type=str, default='dbt+drl', choices=['dae', 'dbt', 'drl', 'dae+dbt', 'dae+drl', 'dbt+drl', 'dae+dbt+drl'], help='training scheme for cycle learning phase')
    parser.add_argument('--noise_type', type=str, default='drop+addition+shuffling', choices=['none', 'drop', 'addition', 'shuffling', 'drop+addition', 'drop+shuffling', 'addition+shuffling', 'drop+addition+shuffling'], help='noisy channels')
    parser.add_argument('--reward_type', type=str, default='flu+sty+rel', choices=['flu', 'sty', 'rel', 'flu+sty', 'flu+rel', 'sty+rel', 'flu+sty+rel'], help='reward types')
    parser.add_argument('--alpha', type=float, default=0.5, help='nl->cf->nl cycle, coefficint combining validity and reconstuction reward')
    parser.add_argument('--beta', type=float, default=0.5, help='cf->nl->cf cycle, coefficint combining validity and reconstuction reward')
    parser.add_argument('--sample_size', type=int, default=4, help='size of sampling during training in dual reinforcement learning')
    parser.add_argument('--train_input_side', type=str, default='nl', choices=['nl', 'cf'], help='type of input utterance for semantic parsing during training')
    parser.add_argument('--eval_input_side', type=str, default='nl', choices=['nl', 'cf'], help='type of input utterance for semantic parsing during evaluation')
    parser.add_argument('--labeled', type=float, default=1.0, help='training use only this propotion of dataset')
    parser.add_argument('--deviceId', type=int, default=0, help='train model on ith gpu. -1: cpu, o.w. gpu index')
    parser.add_argument('--seed', type=int, default=999, help='set initial random seed')
    return parser
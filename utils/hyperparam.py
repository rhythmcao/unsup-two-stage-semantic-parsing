#coding=utf8
''' Construct exp directory according to hyper parameters
'''
import os

EXP_PATH = 'exp'

def hyperparam_path(args):
    if args.read_model_path and args.testing:
        return args.read_model_path
    exp_path = hyperparam_path_base(args)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path

def hyperparam_path_base(args):
    task_path = 'task_%s' % (args.task)
    dataset_path = 'dataset_%s_labeled_%s' % (args.dataset, args.labeled)
    exp_path = ''
    exp_path += 'c_%s__' % (args.cell)
    exp_path += 'es_%s__' % (args.embed_size) if args.pretrained_embed == 'glove' else 'pe_%s__' % (args.pretrained_embed)
    exp_path += 'hd_%s_x_%s__' % (args.hidden_size, args.num_layers)
    exp_path += 'dp_%s__' % (args.dropout)
    exp_path += 'lr_%s__' % (args.lr)
    exp_path += 'l2_%s__' % (args.l2)
    exp_path += 'ld_%s__' % (args.layerwise_decay)
    exp_path += 'sd_%s__' % (args.lr_schedule)
    exp_path += 'mn_%s__' % (args.max_norm)
    exp_path += 'bs_%s__' % (args.batch_size)
    exp_path += 'me_%s__' % (args.max_epoch)
    exp_path += 'bm_%s__' % (args.beam_size)
    exp_path += 'nb_%s' % (args.n_best)
    return os.path.join(EXP_PATH, task_path, dataset_path, exp_path)
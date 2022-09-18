#coding=utf8
'''
    Construct exp directory according to hyper parameters
'''
import os

EXP_PATH = 'exp'

def hyperparam_seq2seq(options):
    task_path = 'task_%s' % (options.task)
    ratio = 'dataset_%s__labeled_%s' % (options.dataset, options.labeled)

    exp_name = 'cell_%s__' % (options.cell)
    exp_name += 'emb_%s__' % (options.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (options.hidden_dim, options.num_layers)
    exp_name += 'trans_%s__' % (options.trans)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s__' % (options.max_epoch)
    exp_name += 'beam_%s__' % (options.beam)
    exp_name += 'nbest_%s' % (options.n_best)
    return os.path.join(EXP_PATH, task_path, ratio, exp_name)

def hyperparam_pretrain_nl2cf_and_cf2nl(options):
    exp_path = hyperparam_seq2seq(options)
    exp_path += '__noisy_%s' % (options.noisy_channel)
    if options.shared_encoder:
        exp_path += '__shared'
    return exp_path

def hyperparam_lm(options):
    task = 'task_%s' % (options.task)
    ratio = 'dataset_%s__labeled_%s' % (options.dataset, options.labeled)

    exp_name = ''
    exp_name += 'cell_%s__' % (options.cell)
    exp_name += 'emb_%s__' % (options.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (options.hidden_dim, options.num_layers)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s' % (options.max_epoch)
    exp_name += '__decTied' if options.decoder_tied else ''
    return os.path.join(EXP_PATH, task, ratio, exp_name)

def hyperparam_classifier(options):
    task = 'task_%s' % (options.task)
    ratio = 'dataset_%s__labeled_%s' % (options.dataset, options.labeled)

    exp_name = ''
    exp_name += 'emb_%s__' % (options.emb_size)
    exp_name += 'filter_'
    for s, n in zip(options.filters, options.filters_num):
        exp_name += '%sx%s_' % (s, n)
    exp_name += '_dropout_%s__' % (options.dropout)
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s' % (options.max_epoch)
    return os.path.join(EXP_PATH, task, ratio, exp_name)

def hyperparam_dual_learning(options):
    task = 'task_%s' % (options.task)
    ratio = 'dataset_%s__labeled_%s' % (options.dataset, options.labeled)

    exp_name = ''
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s__' % (options.max_epoch)
    exp_name += 'beam_%s__' % (options.beam)
    exp_name += 'nbest_%s__' % (options.n_best)
    exp_name += 'sample_%s__alpha_%s__beta_%s__' % (options.sample, options.alpha, options.beta)
    exp_name += 'scheme_%s' % (options.scheme)
    exp_name += '__shared' if options.shared_encoder else ''
    return os.path.join(EXP_PATH, task, ratio, exp_name)
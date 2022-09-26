#coding=utf8
''' Construct exp directory according to hyper parameters
'''
import os

EXP_PATH = 'exp'

def hyperparam_path(args, task='semantic_parsing'):
    if args.read_model_path and args.testing:
        return args.read_model_path
    exp_path = hyperparam_path_dict[task](args)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_sp(args):
    task_path = 'task_%s' % (args.task)
    exp_path = ''
    exp_path += 'cell_%s__' % (args.cell)
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
    exp_path += 'nb_%s__' % (args.n_best)
    exp_path += 'seed_%s' % (args.seed)
    if args.labeled < 1.:
        dataset_path = 'dataset_%s_labeled_%s' % (args.dataset, args.labeled)
        return os.path.join(EXP_PATH, task_path, dataset_path, exp_path)
    else: return os.path.join(EXP_PATH, task_path, 'dataset_%s__' % (args.dataset) + exp_path)


def hyperparam_path_multitask_dae(args):
    task_path = 'task_%s' % (args.task)
    exp_path = ''
    exp_path += 'dataset_%s__' % (args.dataset)
    exp_path += 'noise_%s__' % (args.noise_type)
    exp_path += 'enc_shared__' if args.share_encoder else ''
    exp_path += 'cell_%s__' % (args.cell)
    exp_path += 'es_%s__' % (args.embed_size)
    exp_path += 'hd_%s_x_%s__' % (args.hidden_size, args.num_layers)
    exp_path += 'dp_%s__' % (args.dropout)
    exp_path += 'lr_%s__' % (args.lr)
    exp_path += 'l2_%s__' % (args.l2)
    exp_path += 'mn_%s__' % (args.max_norm)
    exp_path += 'bs_%s__' % (args.batch_size)
    exp_path += 'me_%s__' % (args.max_epoch)
    exp_path += 'bm_%s__' % (args.beam_size)
    exp_path += 'nb_%s__' % (args.n_best)
    exp_path += 'seed_%s' % (args.seed)
    return os.path.join(EXP_PATH, task_path, exp_path)


def hyperparam_path_lm(args):
    task_path = 'task_%s' % (args.task)
    exp_path = 'dataset_%s__' % (args.dataset)
    exp_path += 'cell_%s__' % (args.cell)
    exp_path += 'es_%s__' % (args.embed_size)
    exp_path += 'hd_%s_x_%s__' % (args.hidden_size, args.num_layers)
    exp_path += 'dp_%s__' % (args.dropout)
    exp_path += 'lr_%s__' % (args.lr)
    exp_path += 'l2_%s__' % (args.l2)
    exp_path += 'mn_%s__' % (args.max_norm)
    exp_path += 'bs_%s__' % (args.batch_size)
    exp_path += 'me_%s__' % (args.max_epoch)
    exp_path += 'seed_%s' % (args.seed)
    return os.path.join(EXP_PATH, task_path, exp_path)


def hyperparam_path_tsc(args):
    task_path = 'task_%s' % (args.task)
    exp_path = 'dataset_%s__' % (args.dataset)
    exp_path += 'es_%s__' % (args.embed_size)
    exp_path += 'ft_%s__' % ('+'.join([f'{str(f)}x{str(n)}' for f, n in zip(args.filters, args.filters_num)]))
    exp_path += 'dp_%s__' % (args.dropout)
    exp_path += 'lr_%s__' % (args.lr)
    exp_path += 'l2_%s__' % (args.l2)
    exp_path += 'mn_%s__' % (args.max_norm)
    exp_path += 'bs_%s__' % (args.batch_size)
    exp_path += 'me_%s__' % (args.max_epoch)
    exp_path += 'seed_%s' % (args.seed)
    return os.path.join(EXP_PATH, task_path, exp_path)


def hyperparam_path_cycle(args):
    task_path = 'task_%s' % (args.task)
    dataset_path = 'dataset_%s__labeled_%s' % (args.dataset, args.labeled)
    exp_path = ''
    exp_path += 'scheme_%s__' % (args.train_scheme)
    if 'dae' in args.train_scheme:
        exp_path += 'noise_%s__' % (args.noise_type)
    if 'drl' in args.train_scheme:
        exp_path += 'sample_%s_reward_%s_alpha_%s_beta_%s__' % (args.sample_size, args.reward_type, args.alpha, args.beta)
    exp_path += 'lr_%s__' % (args.lr)
    exp_path += 'l2_%s__' % (args.l2)
    exp_path += 'sd_%s__' % (args.lr_schedule)
    exp_path += 'mn_%s__' % (args.max_norm)
    exp_path += 'bs_%s__' % (args.batch_size)
    exp_path += 'me_%s__' % (args.max_epoch)
    exp_path += 'bm_%s__' % (args.beam_size)
    exp_path += 'nb_%s__' % (args.n_best)
    exp_path += 'seed_%s' % (args.seed)
    return os.path.join(EXP_PATH, task_path, dataset_path, exp_path)


hyperparam_path_dict = {
    'semantic_parsing': hyperparam_path_sp,
    'language_model': hyperparam_path_lm,
    'text_style_classification': hyperparam_path_tsc,
    'multitask_dae': hyperparam_path_multitask_dae,
    'cycle_learning': hyperparam_path_cycle
}

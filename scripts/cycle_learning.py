#coding=utf8
import os, sys, time, json, torch, gc
import numpy as np
import torch.nn as nn
from argparse import Namespace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args import init_args
from utils.constants import PAD
from utils.initialization import initialization_wrapper
from utils.example import Example, split_dataset
from utils.batch import get_minibatch
from utils.optimization import set_optimizer
from models.model_constructor import construct_model
from models.reward_model import RewardModel
from scripts.eval_model import decode, generate_pseudo_dataset

################ initialization ################
args = init_args()
if args.read_model_path: # testing mode
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.read_model_path, params.testing, params.device = args.read_model_path, args.testing, args.device
    params.test_batch_size, params.beam_size, params.n_best = args.test_batch_size, args.beam_size, args.n_best
    params.read_nsp_model_path = args.read_nsp_model_path if args.read_nsp_model_path else params.read_nsp_model_path
    args = params
assert args.read_nsp_model_path is not None and args.read_pdp_model_path is not None
exp_path, logger, device = initialization_wrapper(args, task='cycle_learning')
Example.configuration(args.dataset, args.embed_size, args.noise_type)
train_dataset, dev_dataset = Example.load_dataset(choice='train')
labeled_dataset, _ = split_dataset(train_dataset, split_ratio=args.labeled) # labeled dataset, semi-supervised settings
logger.info("Train and Dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))
test_dataset = Example.load_dataset(choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

################ load auxiliary models ################
if not args.testing:
    reward_model = RewardModel(args.read_language_model_path, args.read_tsc_model_path, args.read_nsp_model_path, device, args.reward_type)
    logger.info(f"Load dual language models from path: {args.read_language_model_path}")
    logger.info(f"Load text style classification model from path: {args.read_tsc_model_path}")
    nsp_model = reward_model.nsp_model
else:
    reward_model = None
    nsp_params = json.load(open(os.path.join(args.read_nsp_model_path, 'params.json'), 'r'))
    nsp_model = construct_model['semantic_parsing'](**nsp_params).to(device)
    check_point = torch.load(open(os.path.join(args.read_nsp_model_path, 'model.pkl'), 'rb'), map_location=device)
    nsp_model.load_state_dict(check_point['model'])
logger.info(f"Load naive semantic parsing model from path: {args.read_nsp_model_path}")

############# load dual paraphrase model #############
if not args.read_model_path:
    paraphrase_params = json.load(open(os.path.join(args.read_pdp_model_path, 'params.json')))
    paraphrase_params['read_nsp_model'], paraphrase_params['read_pdp_model'] = args.read_nsp_model_path, args.read_pdp_model_path
    json.dump(paraphrase_params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else: paraphrase_params = json.load(open(os.path.join(args.read_model_path, 'params.json')))
paraphrase_model = construct_model['dual_paraphrase'](**paraphrase_params)
load_model_path = args.read_model_path if args.read_model_path else args.read_pdp_model_path
check_point = torch.load(open(os.path.join(load_model_path, 'model.pkl'), 'rb'), map_location=torch.device("cpu"))
paraphrase_model.load_state_dict(check_point['model'])
logger.info(f"Load dual paraphrase model from path: {load_model_path}")
model = construct_model['cycle_learning'](paraphrase_model, reward_model, args.alpha, args.beta, args.sample_size).to(device)

decode_task = 'two_stage_semantic_parsing'
################ training/decode ################
if not args.testing:
    loss_function = nn.NLLLoss(ignore_index=Example.vocab.nl2id[PAD], reduction='sum')
    num_training_steps = args.max_epoch * ((len(train_dataset) + args.batch_size - 1) // args.batch_size)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    best_result = {"iter": 0, "nl_bleu": 0., "cf_acc": 0., "dev_acc": 0., "test_acc": 0.}
    logger.info("Start training ... ...")
    if 'dae' in args.train_scheme: dae_nl_index, dae_cf_index = np.arange(len(train_dataset)), np.arange(len(train_dataset))
    if 'dbt' in args.train_scheme: dbt_nl_index, dbt_cf_index = np.arange(len(train_dataset)), np.arange(len(train_dataset))
    if 'drl' in args.train_scheme: drl_nl_index, drl_cf_index = np.arange(len(train_dataset)), np.arange(len(train_dataset))
    if len(labeled_dataset) > 0: labeled_data_index = np.arange(len(labeled_dataset))
    for i in range(args.max_epoch):
        start_time, nl2cf_epoch_loss, cf2nl_epoch_loss = time.time(), 0, 0
        if 'dae' in args.train_scheme: np.random.shuffle(dae_nl_index) ; np.random.shuffle(dae_cf_index)
        if 'dbt' in args.train_scheme: np.random.shuffle(dbt_nl_index) ; np.random.shuffle(dbt_cf_index)
        if 'drl' in args.train_scheme: np.random.shuffle(drl_nl_index) ; np.random.shuffle(drl_cf_index)
        if len(labeled_dataset) > 0: np.random.shuffle(labeled_data_index)

        if 'dbt' in args.train_scheme: # generate pseudo labeled dataset via dual back-translation
            dbt_nl2cf_dataset, dbt_cf2nl_dataset = generate_pseudo_dataset(model, train_dataset, batch_size=args.test_batch_size,
                beam_size=args.beam_size, device=device)
            logger.info(f'Generation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tDual Back-translation dataset construction')
            start_time = time.time()

        model.train()
        for j in range(0, len(train_dataset), args.batch_size):
            optimizer.zero_grad()
            if 'dae' in args.train_scheme:
                # noisy cf -> cf
                inputs, lens, outputs, out_lens = get_minibatch(train_dataset, task='multitask_dae', data_index=dae_cf_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='cf', train_dataset=train_dataset)
                batch_outputs = model(inputs, lens, outputs[:, :-1], task='nl2cf')
                loss = loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
                nl2cf_epoch_loss += loss.item()
                loss.backward()
                # noisy nl -> nl
                inputs, lens, outputs, out_lens = get_minibatch(train_dataset, task='multitask_dae', data_index=dae_nl_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='nl', train_dataset=train_dataset)
                batch_outputs = model(inputs, lens, outputs[:, :-1], task='cf2nl')
                loss = loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
                cf2nl_epoch_loss += loss.item()
                loss.backward()

            if 'dbt' in args.train_scheme:
                # nl -> cf supervised training on pseudo samples
                inputs, lens, outputs, out_lens = get_minibatch(dbt_nl2cf_dataset, task='paraphrase', data_index=dbt_nl_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='nl')
                batch_outputs = model(inputs, lens, outputs[:, :-1], task='nl2cf')
                loss = loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
                nl2cf_epoch_loss += loss.item()
                loss.backward()
                # cf -> nl supervised training on pseudo samples
                inputs, lens, outputs, out_lens = get_minibatch(dbt_cf2nl_dataset, task='paraphrase', data_index=dbt_cf_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='cf')
                batch_outputs = model(inputs, lens, outputs[:, :-1], task='cf2nl')
                loss = loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
                cf2nl_epoch_loss += loss.item()
                loss.backward()

            if 'drl' in args.train_scheme:
                # dual reinforcement learning starts from nl -> cf -> nl
                inputs, lens, raw_inputs, variables = get_minibatch(train_dataset, task='paraphrase_cycle', data_index=drl_nl_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='nl')
                nl2cf_loss, cf2nl_loss = model.cycle_learning(inputs, lens, raw_inputs, variables, task='nl2cf2nl')
                nl2cf_epoch_loss += nl2cf_loss.item()
                cf2nl_epoch_loss += cf2nl_loss.item()
                nl2cf_loss.backward()
                cf2nl_loss.backward()
                # dual reinforcement learning starts from cf -> nl -> cf
                inputs, lens, raw_inputs = get_minibatch(train_dataset, task='paraphrase_cycle', data_index=drl_cf_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='cf')
                cf2nl_loss, nl2cf_loss = model.cycle_learning(inputs, lens, raw_inputs, task='cf2nl2cf')
                nl2cf_epoch_loss += nl2cf_loss.item()
                cf2nl_epoch_loss += cf2nl_loss.item()
                nl2cf_loss.backward()
                cf2nl_loss.backward()

            if len(labeled_dataset) > 0:
                # nl -> cf supervised training
                inputs, lens, outputs, out_lens = get_minibatch(labeled_dataset, task='paraphrase', data_index=labeled_data_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='nl')
                batch_outputs = model(inputs, lens, outputs[:, :-1], task='nl2cf')
                loss = loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
                nl2cf_epoch_loss += loss.item()
                loss.backward()
                # cf -> nl supervised training
                inputs, lens, outputs, out_lens = get_minibatch(labeled_dataset, task='paraphrase', data_index=labeled_data_index,
                    index=j, batch_size=args.batch_size, device=device, input_side='cf')
                batch_outputs = model(inputs, lens, outputs[:, :-1], task='cf2nl')
                loss = loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
                cf2nl_epoch_loss += loss.item()
                loss.backward()

            optimizer.step()
            scheduler.step()

        logger.info(f'Training:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tNL2CF/CF2NL Loss: {nl2cf_epoch_loss:.4f}/{cf2nl_epoch_loss:.4f}')
        gc.collect()
        torch.cuda.empty_cache()

        start_time = time.time()
        nl_bleu, cf_acc = decode(model, dev_dataset, None, args.test_batch_size, beam_size=args.beam_size, device=device, task='unsupervised_cycle_consistency')
        logger.info(f'Unsupervised Dev Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tCycle consistency NL Bleu/CF Acc: {nl_bleu:.4f}/{cf_acc:.4f}')

        start_time = time.time()
        dev_acc = decode(model, nsp_model, dev_dataset, os.path.join(exp_path, 'dev.iter' + str(i)), args.test_batch_size,
            beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
        logger.info(f'Dev Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tDev Acc: {dev_acc:.4f}')

        start_time = time.time()
        test_acc = decode(model, nsp_model, test_dataset, os.path.join(exp_path, 'test.iter' + str(i)), args.test_batch_size,
            beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
        logger.info(f'Test Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tTest Acc: {test_acc:.4f}')

        if nl_bleu + cf_acc > best_result['nl_bleu'] + best_result['cf_acc']:
            best_result['iter'], best_result['nl_bleu'], best_result['cf_acc'], best_result['dev_acc'], best_result['test_acc'] = i, nl_bleu, cf_acc, dev_acc, test_acc
            torch.save({'model': model.paraphrase_model.state_dict(), 'result': best_result}, open(os.path.join(exp_path, 'model.pkl'), 'wb'))
            logger.info(f'NEW BEST:\tEpoch: {i:d}\tBest Dev Acc: {dev_acc:.4f}\tBest Test Acc: {test_acc:.4f}')

    logger.info(f"FINAL BEST:\tEpoch: {best_result['iter']:d}\tBest Dev Acc: {best_result['dev_acc']:.4f}\tBest Test Acc: {best_result['test_acc']:.4f}")
else:
    logger.info("Start evaluating ... ...")
    start_time = time.time()
    dev_acc = decode(model, nsp_model, dev_dataset, os.path.join(exp_path, 'dev.eval'), args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
    logger.info(f'Evaluation cost: {time.time() - start_time:.2f}s\tDev Acc: {dev_acc:.4f}')

    start_time = time.time()
    test_acc = decode(model, nsp_model, test_dataset, os.path.join(exp_path, 'test.eval'), args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
    logger.info(f'Evaluation cost: {time.time() - start_time:.2f}s\tTest Acc: {test_acc:.4f}')

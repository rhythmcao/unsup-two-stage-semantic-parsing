#coding=utf8
import os, sys, time, json, torch, gc
import numpy as np
import torch.nn as nn
from argparse import Namespace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args import init_args
from utils.constants import PAD
from utils.initialization import initialization_wrapper
from utils.example import Example
from utils.batch import get_minibatch
from utils.optimization import set_optimizer
from models.model_constructor import construct_model
from scripts.eval_model import decode

################ initialization ################
args = init_args()
if args.read_model_path: # testing mode
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.read_model_path, params.testing, params.device = args.read_model_path, args.testing, args.device
    params.test_batch_size, params.beam_size, params.n_best = args.test_batch_size, args.beam_size, args.n_best
    args = params
exp_path, logger, device = initialization_wrapper(args, task='multitask_dae')
Example.configuration(args.dataset, args.embed_size, args.noise_type)
train_dataset, dev_dataset = Example.load_dataset(choice='train')
logger.info("Train and Dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))
test_dataset = Example.load_dataset(choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

################ construct model ################
if not args.read_model_path:
    args.src_vocab_size, args.tgt_vocab_size = len(Example.vocab.nl2id), len(Example.vocab.lf2id)
    args.src_pad_idx, args.tgt_pad_idx = Example.vocab.nl2id[PAD], Example.vocab.lf2id[PAD]
    json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
model = construct_model['multitask_semantic_parsing'](**vars(args)).to(device)

############# model initialization #############
if not args.read_model_path: # init word embeddings with GloVe
    ratio = Example.word2vec.load_embeddings(model.sp_model.src_embed.embed, Example.vocab.nl2id, device)
    logger.info(f"{ratio * 100:.2f} word embeddings initialized from GloVe")
else:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.pkl'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load model from path: {args.read_model_path}")

################ training/decode ################
decode_task = 'one_stage_semantic_parsing'
if not args.testing:
    sp_loss_function = nn.NLLLoss(ignore_index=Example.vocab.lf2id[PAD], reduction='sum')
    dae_loss_function = nn.NLLLoss(ignore_index=Example.vocab.nl2id[PAD], reduction='sum')
    num_training_steps = args.max_epoch * ((len(train_dataset) + args.batch_size - 1) // args.batch_size)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    best_result = {"iter": 0, "dev_acc": 0., "test_acc": 0.}
    logger.info("Start training ... ...")
    train_data_index, dae_data_index = np.arange(len(train_dataset)), np.arange(len(train_dataset))

    for i in range(args.max_epoch):
        start_time, sp_epoch_loss, dae_epoch_loss = time.time(), 0, 0
        np.random.shuffle(train_data_index)
        np.random.shuffle(dae_data_index)
        model.train()
        for j in range(0, len(train_dataset), args.batch_size):
            optimizer.zero_grad()
            # semantic parsing cf -> lf
            inputs, lens, outputs, out_lens = get_minibatch(train_dataset, task='semantic_parsing', data_index=train_data_index,
                index=j, batch_size=args.batch_size, device=device, input_side='cf')
            batch_outputs = model(inputs, lens, outputs[:, :-1], task='semantic_parsing')
            loss = sp_loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.lf2id)), outputs[:, 1:].contiguous().view(-1))
            sp_epoch_loss += loss.item()
            loss.backward()
            # multitask DAE, noisy nl -> nl
            inputs, lens, outputs, out_lens = get_minibatch(train_dataset, task='multitask_dae', data_index=dae_data_index,
                index=j, batch_size=args.batch_size, device=device, input_side='nl', train_dataset=train_dataset)
            batch_outputs = model(inputs, lens, outputs[:, :-1], task='multitask_dae')
            loss = dae_loss_function(batch_outputs.contiguous().view(-1, len(Example.vocab.nl2id)), outputs[:, 1:].contiguous().view(-1))
            dae_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.info(f'Training:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tSP/DAE Loss: {sp_epoch_loss:.4f}/{dae_epoch_loss:.4f}')
        gc.collect()
        torch.cuda.empty_cache()
        if i < args.eval_after_epoch:
            continue

        start_time = time.time()
        dev_acc = decode(model, dev_dataset, os.path.join(exp_path, f'dev.iter{i:d}'), args.test_batch_size,
            beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
        logger.info(f'Dev Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tDev Acc: {dev_acc:.4f}')

        start_time = time.time()
        test_acc = decode(model, test_dataset, os.path.join(exp_path, f'test.iter{i:d}'), args.test_batch_size,
            beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
        logger.info(f'Test Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tTest Acc: {test_acc:.4f}')

        if dev_acc > best_result['dev_acc']:
            best_result['iter'], best_result['dev_acc'], best_result['test_acc'] = i, dev_acc, test_acc
            torch.save({'model': model.state_dict(), 'result': best_result}, open(os.path.join(exp_path, 'model.pkl'), 'wb'))
            logger.info(f'NEW BEST:\tEpoch: {i:d}\tBest Dev Acc: {dev_acc:.4f}\tBest Test Acc: {test_acc:.4f}')

    logger.info(f"FINAL BEST:\tEpoch: {best_result['iter']:d}\tBest Dev Acc: {best_result['dev_acc']:.4f}\tBest Test Acc: {best_result['test_acc']:.4f}")
else:
    logger.info("Start evaluating ... ...")
    start_time = time.time()
    dev_acc = decode(model, dev_dataset, os.path.join(exp_path, 'dev.eval'), args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
    logger.info(f'Evaluation cost: {time.time() - start_time:.2f}s\tDev Acc: {dev_acc:.4f}')

    start_time = time.time()
    test_acc = decode(model, test_dataset, os.path.join(exp_path, 'test.eval'), args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, device=device, task=decode_task)
    logger.info(f'Evaluation cost: {time.time() - start_time:.2f}s\tTest Acc: {test_acc:.4f}')

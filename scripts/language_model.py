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
exp_path, logger, device = initialization_wrapper(args, task='language_model')
Example.configuration(args.dataset, args.embed_size)
train_dataset, dev_dataset = Example.load_dataset(choice='train')
logger.info("Train and Dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))

################ construct model ################
if not args.read_model_path:
    args.vocab_size, args.pad_idx = len(Example.vocab.nl2id), Example.vocab.nl2id[PAD]
    json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
model = construct_model['language_model'](**vars(args)).to(device)

############# model initialization #############
if not args.read_model_path: # init word embeddings with GloVe
    ratio = Example.word2vec.load_embeddings(model.nl_lm.encoder, Example.vocab.nl2id, device)
    logger.info(f"{ratio * 100:.2f} word embeddings initialized from GloVe")
else:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.pkl'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load model from path: {args.read_model_path}")

decode_task = 'language_model'
################ training/decode ################
if not args.testing:
    loss_function = nn.NLLLoss(ignore_index=Example.vocab.nl2id[PAD], reduction='sum')
    num_training_steps = args.max_epoch * ((len(train_dataset) + args.batch_size - 1) // args.batch_size)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    optimizer, scheduler = set_optimizer(model, args, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    best_result = {"iter": 0, "nl_ppl": 1e8, "cf_ppl": 1e8}
    logger.info("Start training ... ...")
    train_data_index = np.arange(len(train_dataset))

    for i in range(args.max_epoch):
        start_time, nl_epoch_loss, cf_epoch_loss = time.time(), 0, 0
        np.random.shuffle(train_data_index)
        model.train()
        for j in range(0, len(train_dataset), args.batch_size):
            optimizer.zero_grad()
            inputs, lens = get_minibatch(train_dataset, task='language_model', data_index=train_data_index,
                index=j, batch_size=args.batch_size, device=device, input_side='nl')
            batch_outputs = model(inputs, lens, input_side='nl')
            loss = loss_function(batch_outputs.contiguous().view(-1, args.vocab_size), inputs[:, 1:].contiguous().view(-1))
            nl_epoch_loss += loss.item()
            loss.backward()
            inputs, lens = get_minibatch(train_dataset, task='language_model', data_index=train_data_index,
                index=j, batch_size=args.batch_size, device=device, input_side='cf')
            batch_outputs = model(inputs, lens, input_side='cf')
            loss = loss_function(batch_outputs.contiguous().view(-1, args.vocab_size), inputs[:, 1:].contiguous().view(-1))
            cf_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.info(f'Training:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tNL/CF Loss: {nl_epoch_loss:.4f}/{cf_epoch_loss:.4f}')
        gc.collect()
        torch.cuda.empty_cache()
        if i < args.eval_after_epoch:
            continue

        start_time = time.time()
        nl_ppl, cf_ppl = decode(model, dev_dataset, None, args.test_batch_size, device=device, task=decode_task)
        logger.info(f'Dev Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tNL/CF PPL: {nl_ppl:.4f}/{cf_ppl:.4f}')

        if nl_ppl + cf_ppl < best_result['nl_ppl'] + best_result['cf_ppl']:
            best_result['iter'], best_result['nl_ppl'], best_result['cf_ppl'] = i, nl_ppl, cf_ppl
            torch.save({'model': model.state_dict(), 'result': best_result}, open(os.path.join(exp_path, 'model.pkl'), 'wb'))
            logger.info(f'NEW BEST:\tEpoch: {i:d}\tBest NL/CF PPL: {nl_ppl:.4f}/{cf_ppl:.4f}')

    logger.info(f"FINAL BEST:\tEpoch: {best_result['iter']:d}\tBest NL/CF PPL: {best_result['nl_ppl']:.4f}/{best_result['cf_ppl']:.4f}")
else:
    logger.info("Start evaluating ... ...")
    start_time = time.time()
    nl_ppl, cf_ppl = decode(model, dev_dataset, None, args.test_batch_size, device=device, task=decode_task)
    logger.info(f'Evaluation cost: {time.time() - start_time:.2f}s\tNL/CF PPL: {nl_ppl:.4f}/{cf_ppl:.4f}')
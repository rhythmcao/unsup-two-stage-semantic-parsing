#coding=utf8
import os, sys, time, json, torch, gc
import numpy as np
import torch.nn as nn
from argparse import Namespace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args import init_args
from utils.constants import PAD
from utils.initialization import initialization_wrapper
from utils.example import Example, UtteranceExample
from utils.batch import get_minibatch
from utils.optimization import set_optimizer_adadelta
from models.model_constructor import construct_model
from scripts.eval_model import decode

################ initialization ################
args = init_args()
if args.read_model_path: # testing mode
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.read_model_path, params.testing, params.device = args.read_model_path, args.testing, args.device
    params.test_batch_size, params.beam_size, params.n_best = args.test_batch_size, args.beam_size, args.n_best
    args = params
exp_path, logger, device = initialization_wrapper(args, task='text_style_classification')
Example.configuration(args.dataset, args.embed_size)
train_dataset, dev_dataset = Example.load_dataset(choice='train')
train_dataset, dev_dataset = UtteranceExample.from_dataset(train_dataset), UtteranceExample.from_dataset(dev_dataset)
logger.info("Train and Dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))

################ construct model ################
if not args.read_model_path:
    args.vocab_size, args.pad_idx = len(Example.vocab.nl2id), Example.vocab.nl2id[PAD]
    json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
model = construct_model['text_style_classification'](**vars(args)).to(device)

############# model initialization #############
if not args.read_model_path: # init word embeddings with GloVe
    ratio = Example.word2vec.load_embeddings(model.embedding, Example.vocab.nl2id, device)
    logger.info(f"{ratio * 100:.2f} word embeddings initialized from GloVe")
else:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.pkl'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load model from path: {args.read_model_path}")

decode_task = 'text_style_classification'
################ training/decode ################
if not args.testing:
    loss_function = nn.BCELoss(reduction='sum')
    optimizer = set_optimizer_adadelta(model, args)
    best_result = {"iter": 0, "dev_acc": 0}
    logger.info("Start training ... ...")
    train_data_index = np.arange(len(train_dataset))

    for i in range(args.max_epoch):
        start_time, epoch_loss = time.time(), 0.
        np.random.shuffle(train_data_index)
        model.train()
        for j in range(0, len(train_dataset), args.batch_size):
            optimizer.zero_grad()
            inputs, outputs = get_minibatch(train_dataset, task='text_style_classification', data_index=train_data_index,
                index=j, batch_size=args.batch_size, device=device)
            batch_outputs = model(inputs)
            loss = loss_function(batch_outputs, outputs)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        logger.info(f'Training:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tEpoch Loss: {epoch_loss:.4f}')
        gc.collect()
        torch.cuda.empty_cache()
        if i < args.eval_after_epoch:
            continue

        start_time = time.time()
        dev_acc = decode(model, dev_dataset, None, args.test_batch_size, device=device, task=decode_task)
        logger.info(f'Dev Evaluation:\tEpoch: {i:d}\tTime: {time.time() - start_time:.2f}s\tDev acc: {dev_acc:.4f}')

        if dev_acc > best_result['dev_acc']:
            best_result['iter'], best_result['dev_acc'] = i, dev_acc
            torch.save({'model': model.state_dict(), 'result': best_result}, open(os.path.join(exp_path, 'model.pkl'), 'wb'))
            logger.info(f'NEW BEST:\tEpoch: {i:d}\tBest Dev Acc: {dev_acc:.4f}')

    logger.info(f"FINAL BEST:\tEpoch: {best_result['iter']:d}\tBest Dev Acc: {best_result['dev_acc']:.4f}")
else:
    logger.info("Start evaluating ... ...")
    start_time = time.time()
    dev_acc = decode(model, dev_dataset, None, args.test_batch_size, device=device, task=decode_task)
    logger.info(f'Evaluation cost: {time.time() - start_time:.2f}s\tDev Acc: {dev_acc:.4f}')
#coding=utf8
import sys, os, random
import torch
from utils.constants import *
from utils.example import Example

def get_minibatch(data_list, vocab, task='nl2cf', data_index=None, index=0, batch_size=16, device=None, **kargs):
    index = index % len(data_list)
    batch_data_list = [data_list[idx] for idx in data_index[index: index + batch_size]]
    return BATCH_FUNC[task](batch_data_list, vocab, device, **kargs)

def get_minibatch_sp(ex_list, vocab, device, cf_vocab=True, cf_input=False, **kargs):
    choice = 'cf' if cf_input else 'nl'
    vocab_choice = 'cf' if cf_vocab else 'nl'
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, choice, vocab_choice, device=device)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'lf', 'lf', True, device=device)
    variables = [ex.variables for ex in ex_list]
    return inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs, variables)

def get_minibatch_pem(ex_list, vocab, device, cf_input=False, **kargs):
    choice = 'cf' if cf_input else 'nl'
    _, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, choice, 'nl', device=device)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'lf', 'lf', True, device=device)
    variables = [ex.variables for ex in ex_list]
    return raw_inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs, variables)

def get_minibatch_nl2cf(ex_list, vocab, device, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'nl', 'nl', device=device)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'cf', 'cf', True, device=device)
    raw_lfs = [ex.lf if hasattr(ex, 'lf') else [] for ex in ex_list]
    variables = [ex.variables for ex in ex_list]
    return inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs, raw_lfs, variables)

def get_minibatch_cf2nl(ex_list, vocab, device, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'cf', 'cf', device=device)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'nl', 'nl', True, device=device)
    return inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs)

def get_minibatch_unlabeled_nl2cf(ex_list, vocab, device, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'nl', 'nl', device=device)
    return inputs, lens, raw_inputs

def get_minibatch_unlabeled_cf2nl(ex_list, vocab, device, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'cf', 'cf', device=device)
    raw_lf_inputs = [ex.lf if hasattr(ex, 'lf') else [] for ex in ex_list]
    variables = [ex.variables for ex in ex_list]
    return inputs, lens, (raw_inputs, raw_lf_inputs, variables)

def get_minibatch_self_train_nl2cf(ex_list, vocab, device, train_dataset=None, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'cf', 'nl', device=device, noisy=True, train_dataset=train_dataset)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'cf', 'cf', True, device=device)
    raw_lfs = [ex.lf if hasattr(ex, 'lf') else [] for ex in ex_list]
    return inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs, raw_lfs)

def get_minibatch_self_train_cf2nl(ex_list, vocab, device, train_dataset=None, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'nl', 'cf', device=device, noisy=True, train_dataset=train_dataset)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'nl', 'nl', True, device=device)
    return inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs)

def get_minibatch_lm(ex_list, vocab, device, side='nl', **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, side, side, True, device=device)
    return inputs, lens, raw_inputs

def get_minibatch_mtsp(ex_list, vocab, device, train_dataset=None, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'nl', 'nl', device=device, noisy=True, train_dataset=train_dataset)
    outputs, out_lens, raw_outputs = get_minibatch_atom(ex_list, vocab, 'nl', 'cf', True, device=device)
    return inputs, lens, outputs, out_lens, (raw_inputs, raw_outputs)

def get_minibatch_dis(ex_list, vocab, device, **kargs):
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, vocab, 'nl', 'nl', device=device)
    labels = [ex.label for ex in ex_list]
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    return inputs, labels, raw_inputs

def get_minibatch_atom(ex_list, vocab, choice='cf', vocab_choice='cf', bos_eos=False, device='cpu', noisy=False, train_dataset=None):
    assert choice in ['nl', 'cf', 'lf'] and vocab_choice in ['nl', 'cf', 'lf']
    inputs = [eval('ex.' + choice) for ex in ex_list]
    if noisy:
        noise_obj = Example.noise
        inputs = [noise_obj.noisy_channel(sent, side=choice, train_dataset=train_dataset) for sent in inputs]
    bos_eos_inputs = [[BOS] + sent + [EOS] for sent in inputs] if bos_eos else inputs
    lens = [len(sent) for sent in bos_eos_inputs]
    max_len = max(lens)
    word2id = eval('vocab.' + vocab_choice + '2id')
    inputs_idx = [[word2id.get(w, word2id[UNK]) for w in sent] for sent in bos_eos_inputs]
    padded_inputs_idx = [sent + [word2id[PAD]] * (max_len - len(sent)) for sent in inputs_idx]
    inputs_tensor = torch.tensor(padded_inputs_idx, dtype=torch.long, device=device)
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)
    return inputs_tensor, lens_tensor, inputs

BATCH_FUNC = {
    "semantic_parsing": get_minibatch_sp,
    "pretrained_embed_model": get_minibatch_pem,
    "nl2cf": get_minibatch_nl2cf,
    "cf2nl": get_minibatch_cf2nl,
    "unlabeled_nl2cf": get_minibatch_unlabeled_nl2cf,
    "unlabeled_cf2nl": get_minibatch_unlabeled_cf2nl,
    "self_train_nl2cf": get_minibatch_self_train_nl2cf,
    "self_train_cf2nl": get_minibatch_self_train_cf2nl,
    "multi_task": get_minibatch_mtsp,
    "language_model": get_minibatch_lm,
    "discriminator": get_minibatch_dis
}

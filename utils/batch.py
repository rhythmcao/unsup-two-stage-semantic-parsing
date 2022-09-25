#coding=utf8
import torch
from utils.constants import *
from utils.example import Example


def get_minibatch(ex_list, task='semantic_parsing', data_index=None, index=0, batch_size=16, device=None, **kwargs):
    index = index % len(ex_list)
    ex_list = [ex_list[idx] for idx in data_index[index: index + batch_size]]
    return BATCH_FUNC[task](ex_list, device, **kwargs)


def get_minibatch_semantic_parsing(ex_list, device='cpu', input_side='nl', labeled=True, pretrained_embed='glove', **kwargs):
    assert input_side in ['nl', 'cf']
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, input_side=input_side, device=device)
    if pretrained_embed in ['elmo', 'bert']: inputs = raw_inputs
    if labeled:
        outputs, out_lens, _ = get_minibatch_atom(ex_list, input_side='lf', bos_eos=True, device=device)
        return inputs, lens, outputs, out_lens
    else: return inputs, lens


def get_minibatch_paraphrase(ex_list, device='cpu', input_side='nl', labeled=True, **kwargs):
    assert input_side in ['nl', 'cf']
    inputs, lens, _ = get_minibatch_atom(ex_list, input_side=input_side, device=device)
    if labeled:
        output_side = 'cf' if input_side == 'nl' else 'nl'
        outputs, out_lens, _ = get_minibatch_atom(ex_list, input_side=output_side, bos_eos=True, device=device)
        return inputs, lens, outputs, out_lens
    else: return inputs, lens


def get_minibatch_paraphrase_cycle(ex_list, device='cpu', input_side='nl', **kwargs):
    assert input_side in ['nl', 'cf']
    inputs, lens, raw_inputs = get_minibatch_atom(ex_list, input_side=input_side, device=device)
    if input_side == 'nl': # need additional grammar_check validity reward for intermediate CF
        variables = [ex.variables for ex in ex_list]
        return inputs, lens, raw_inputs, variables
    return inputs, lens, raw_inputs


def get_minibatch_language_model(ex_list, device='cpu', input_side='nl'):
    assert input_side in ['nl', 'cf']
    inputs, lens, _ = get_minibatch_atom(ex_list, input_side=input_side, bos_eos=True, device=device)
    return inputs, lens


def get_minibatch_text_style_classification(ex_list, device='cpu', labeled=True, **kwargs):
    inputs, _, _ = get_minibatch_atom(ex_list, input_side='nl', device=device)
    if labeled:
        labels = torch.tensor([ex.label for ex in ex_list], dtype=torch.float, device=device)
        return inputs, labels
    return inputs


def get_minibatch_multitask_dae(ex_list, device='cpu', input_side='nl', train_dataset=[], **kwargs):
    inputs, lens, _ = get_minibatch_atom(ex_list, input_side=input_side, device=device, noisy=True, train_dataset=train_dataset)
    outputs, out_lens, _ = get_minibatch_atom(ex_list, input_side=input_side, bos_eos=True, device=device)
    return inputs, lens, outputs, out_lens


def get_minibatch_atom(ex_list, input_side='nl', bos_eos=False, device='cpu', noisy=False, train_dataset=None):
    assert input_side in ['nl', 'cf', 'lf']
    inputs = [eval('ex.' + input_side) for ex in ex_list]
    if noisy:
        noise = Example.noise
        inputs = [noise.noisy_channel(sent, input_side=input_side, train_dataset=train_dataset) for sent in inputs]
    bos_eos_inputs = [[BOS] + sent + [EOS] for sent in inputs] if bos_eos else inputs
    lens = [len(sent) for sent in bos_eos_inputs]
    max_len = max(lens)
    word2id = Example.vocab.nl2id if input_side != 'lf' else Example.vocab.lf2id
    inputs_idx = [[word2id.get(w, word2id[UNK]) for w in sent] for sent in bos_eos_inputs]
    padded_inputs_idx = [sent + [word2id[PAD]] * (max_len - len(sent)) for sent in inputs_idx]
    inputs_tensor = torch.tensor(padded_inputs_idx, dtype=torch.long, device=device)
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)
    return inputs_tensor, lens_tensor, inputs


BATCH_FUNC = {
    "semantic_parsing": get_minibatch_semantic_parsing,
    "paraphrase": get_minibatch_paraphrase,
    "paraphrase_cycle": get_minibatch_paraphrase_cycle,
    "language_model": get_minibatch_language_model,
    "text_style_classification": get_minibatch_text_style_classification,
    "multitask_dae": get_minibatch_multitask_dae
}
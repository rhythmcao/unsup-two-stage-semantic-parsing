#coding=utf8
import torch
import torch.nn as nn
import numpy as np
from utils.constants import PAD
from utils.example import Example
from utils.batch import get_minibatch


def generate_pseudo_dataset(model, dataset, batch_size=128, beam_size=5, device='cpu'):
    model.eval()
    data_index = np.arange(len(dataset))
    domain, vocab = Example.domain, Example.vocab
    nl_predictions, cf_predictions = [], []
    with torch.no_grad():
        for j in range(0, len(dataset), batch_size):
            inputs, lens = get_minibatch(dataset, task='paraphrase', data_index=data_index, index=j,
                batch_size=batch_size, device=device, input_side='nl', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1, task='nl2cf')
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            cf_predictions.extend(preds)

            inputs, lens = get_minibatch(dataset, task='paraphrase', data_index=data_index, index=j,
                batch_size=batch_size, device=device, input_side='cf', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1, task='cf2nl')
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            nl_predictions.extend(preds)
    nl_references, cf_references = [ex.nl for ex in dataset], [ex.cf for ex in dataset]
    pseudo_nl2cf_dataset = list(map(lambda ex: Example(nl=ex[0], cf=ex[1]), zip(nl_predictions, cf_references)))
    pseudo_cf2nl_dataset = list(map(lambda ex: Example(nl=ex[1], cf=ex[0]), zip(cf_predictions, nl_references)))
    return pseudo_nl2cf_dataset, pseudo_cf2nl_dataset


def decode(*args, **kwargs):
    task = kwargs.pop('task', 'one_stage_semantic_parsing')
    return DECODE_FUNC[task](*args, **kwargs)


def decode_one_stage_semantic_parsing(model, dataset, output_path=None, batch_size=128, beam_size=5, n_best=1, input_side='nl', pretrained_embed='glove', device='cpu', *kwargs):
    model.eval()
    data_index = np.arange(len(dataset))
    domain, vocab = Example.domain, Example.vocab
    predictions, references, variables = [], [ex.lf for ex in dataset], [ex.variables for ex in dataset]
    with torch.no_grad():
        for j in range(0, len(dataset), batch_size):
            inputs, lens = get_minibatch(dataset, task='semantic_parsing', data_index=data_index, index=j,
                batch_size=batch_size, device=device, input_side=input_side, labeled=False, pretrained_embed=pretrained_embed)
            results = model.decode_batch(inputs, lens, vocab.lf2id, beam_size=beam_size, n_best=n_best)
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2lf)
            predictions.extend(preds)
    scores = domain.compare_lf(predictions, references, variables=variables)
    accuracy = sum(scores) / float(len(scores))

    if output_path is not None:
        with open(output_path, 'w') as of:
            for idx, ex in enumerate(dataset):
                if int(scores[idx]) == 0:
                    if input_side == 'nl': of.write("Input NL: " + ' '.join(ex.nl) + '\n')
                    else: of.write("Input CF: " + ' '.join(ex.cf) + '\n')
                    of.write("Gold LF: " + ' '.join(ex.lf) + '\n')
                    for jdx in range(n_best):
                        of.write("Pred" + str(jdx) + " LF: " + ' '.join(predictions[n_best * idx + jdx]) + '\n')
                    of.write('\n')
            of.write(f'Overall accuracy is {accuracy:.4f} .')
    return accuracy


def decode_two_stage_semantic_parsing(model, nsp_model, dataset, output_path=None, batch_size=128, beam_size=5, n_best=1, device='cpu', *kwargs):
    model.eval()
    nsp_model.eval()
    domain, vocab = Example.domain, Example.vocab
    data_index = np.arange(len(dataset))
    with torch.no_grad():
        cf_predictions, lf_predictions = [], []
        for j in range(0, len(dataset), batch_size):
            inputs, lens = get_minibatch(dataset, task='paraphrase',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='nl', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1)
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            cf_predictions.extend(preds)

        pred_dataset = list(map(lambda cf: Example(cf=cf), cf_predictions))
        for j in range(0, len(pred_dataset), batch_size):
            inputs, lens = get_minibatch(pred_dataset, task='semantic_parsing',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='cf', labeled=False)
            results = nsp_model.decode_batch(inputs, lens, vocab.lf2id, beam_size=beam_size, n_best=n_best)
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2lf)
            lf_predictions.extend(preds)
    references, variables = [ex.lf for ex in dataset], [ex.variables for ex in dataset]
    scores = domain.compare_lf(lf_predictions, references, variables=variables)
    accuracy = sum(scores) / float(len(scores))

    if output_path is not None:
        with open(output_path, 'w') as of:
            for idx, ex in enumerate(dataset):
                if int(scores[idx]) == 0:
                    of.write("Input NL: " + ' '.join(ex.nl) + '\n')
                    of.write("Gold CF: " + ' '.join(ex.cf) + '\n')
                    of.write("Pred CF:" + ' '.join(pred_dataset[idx].cf) + '\n')
                    of.write("Gold LF: " + ' '.join(ex.lf) + '\n')
                    for jdx in range(n_best):
                        of.write("Pred" + str(jdx) + " LF: " + ' '.join(lf_predictions[n_best * idx + jdx]) + '\n')
                    of.write('\n')
            of.write(f'Overall accuracy is {accuracy:.4f} .')
    return accuracy


def decode_language_model(model, dataset, output_path=None, batch_size=128, device='cpu', **kwargs):
    model.eval()
    data_index = np.arange(len(dataset))
    nll_score = nn.NLLLoss(ignore_index=Example.vocab.nl2id[PAD], reduction='sum')
    nl_lens, nl_nll, cf_lens, cf_nll = 0, [], 0, []
    with torch.no_grad():
        for j in range(0, len(dataset), batch_size):
            inputs, lens = get_minibatch(dataset, task='language_model',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='nl')
            outputs = model(inputs, lens, input_side='nl')
            score = nll_score(outputs.contiguous().view(-1, len(Example.vocab.nl2id)), inputs[:, 1:].contiguous().view(-1))
            nl_nll.append(score.item())
            nl_lens += torch.sum(lens - 1).item()

            inputs, lens = get_minibatch(dataset, task='language_model',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='cf')
            outputs = model(inputs, lens, input_side='cf')
            score = nll_score(outputs.contiguous().view(-1, len(Example.vocab.nl2id)), inputs[:, 1:].contiguous().view(-1))
            cf_nll.append(score.item())
            cf_lens += torch.sum(lens - 1).item()

    nl_ppl = np.exp(np.sum(nl_nll, axis=0) / nl_lens)
    cf_ppl = np.exp(np.sum(cf_nll, axis=0) / cf_lens)
    return nl_ppl, cf_ppl


def decode_text_style_classification(model, dataset, output_path=None, batch_size=128, device='cpu', **kwargs):
    model.eval()
    correct, data_index = 0, np.arange(len(dataset))
    with torch.no_grad():
        for j in range(0, len(dataset), batch_size):
            inputs, outputs = get_minibatch(dataset, task='text_style_classification',
                data_index=data_index, index=j, batch_size=batch_size, device=device)
            batch_outputs = model(inputs)
            correct += torch.sum(~ torch.logical_xor(batch_outputs >= 0.5, outputs.bool())).item()
    accuracy = correct / len(dataset)
    return accuracy


def decode_unsupervised_cycle_consistency(model, dataset, output_path=None, batch_size=128, beam_size=5, n_best=1, device='cpu', **kwargs):
    # nl -> cf -> nl, calculate reconstruction nl bleu score
    # cf -> nl -> cf, calculate reconstruction cf sentence accuracy
    model.eval()
    domain, vocab = Example.domain, Example.vocab
    data_index = np.arange(len(dataset))
    with torch.no_grad():
        nl_predictions, cf_predictions = [], []
        for j in range(0, len(dataset), batch_size):
            inputs, lens = get_minibatch(dataset, task='paraphrase',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='nl', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1, task='nl2cf')
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            cf_predictions.extend(preds)

            inputs, lens = get_minibatch(dataset, task='paraphrase',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='cf', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1, task='cf2nl')
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            nl_predictions.extend(preds)

        nl_pred_dataset = list(map(lambda nl: Example(nl=nl), nl_predictions))
        cf_pred_dataset = list(map(lambda cf: Example(cf=cf), cf_predictions))
        nl_predictions, cf_predictions = [], []
        for j in range(0, len(dataset), batch_size):
            inputs, lens = get_minibatch(nl_pred_dataset, task='paraphrase',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='nl', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1, task='nl2cf')
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            cf_predictions.extend(preds)

            inputs, lens = get_minibatch(cf_pred_dataset, task='paraphrase',
                data_index=data_index, index=j, batch_size=batch_size, device=device, input_side='cf', labeled=False)
            results = model.decode_batch(inputs, lens, vocab.nl2id, beam_size=beam_size, n_best=1, task='cf2nl')
            preds = domain.reverse(sum(results["predictions"], []), vocab.id2nl)
            nl_predictions.extend(preds)

    nl_references, cf_references = [ex.nl for ex in dataset], [ex.cf for ex in dataset]
    bleu_list, acc_list = domain.compare_nl(nl_predictions, nl_references), domain.compare_cf(cf_predictions, cf_references)
    avg_bleu, avg_acc = sum(bleu_list) / len(bleu_list), sum(acc_list) / len(acc_list)
    return avg_bleu, avg_acc


DECODE_FUNC = {
    'one_stage_semantic_parsing': decode_one_stage_semantic_parsing,
    'two_stage_semantic_parsing': decode_two_stage_semantic_parsing,
    'unsupervised_cycle_consistency': decode_unsupervised_cycle_consistency,
    'language_model': decode_language_model,
    'text_style_classification': decode_text_style_classification,
}
#coding=utf8
import torch
import torch.nn as nn
from utils.example import Example
from utils.batch import get_minibatch_paraphrase


class CycleLearningModel(nn.Module):

    def __init__(self, paraphrase_model, reward_model, alpha=0.5, beta=0.5, sample_size=6, **kargs):
        """
        @args:
            alpha: reward for cycle nl2cf2nl = cf_val_reward * alpha + nl_rec_reward * (1 - alpha)
            beta: reward for cycle cf2nl2cf = nl_val_reward * beta + cf_rec_reward * (1 - beta)
            sample_size: beam search and sample size for training in cycle learning cycles
        """
        super(CycleLearningModel, self).__init__()
        self.paraphrase_model = paraphrase_model
        self.reward_model = reward_model # not class nn.Module
        self.alpha, self.beta, self.sample_size = alpha, beta, sample_size


    def forward(self, *args, **kwargs):
        """ Directly invoke the paraphrase model
        """
        return self.paraphrase_model(*args, **kwargs)


    def decode_batch(self, *args, **kwargs):
        """ Directly invoke the paraphrase model
        """
        return self.paraphrase_model.decode_batch(*args, **kwargs)


    def cycle_learning(self, inputs, lens, raw_inputs, variables=[], task='nl2cf2nl'):
        assert task in ['nl2cf2nl', 'cf2nl2cf']
        domain, vocab = Example.domain, Example.vocab
        if task == 'nl2cf2nl': primal_task, dual_task, coefficient = task[:5], task[3:], self.alpha
        else: primal_task, dual_task, coefficient = task[:5], task[3:], self.beta
        # primal model
        results = self.paraphrase_model.decode_batch(inputs, lens, vocab.nl2id, beam_size=self.sample_size, n_best=self.sample_size, task=primal_task)
        predictions, primal_scores = results['predictions'], results['scores']
        predictions = sum(predictions, [])
        predictions = domain.reverse(predictions, vocab.id2nl)
        raw_inputs = [each for each in raw_inputs for _ in range(self.sample_size)] # repeat sample_size times

        # calculate validity reward
        val_reward = self.reward_model(predictions, variables, choice=task + '_val').contiguous().view(-1, self.sample_size).to(primal_scores.device)
        baseline = val_reward.mean(dim=-1, keepdim=True) # baseline is the average reward in the beam
        val_reward = val_reward - baseline

        # dual model
        rec_inputs, rec_lens, rec_outputs, rec_out_lens = self.construct_reverse_input(predictions, raw_inputs, dual_task, device=primal_scores.device)
        logscore = self.paraphrase_model(rec_inputs, rec_lens, rec_outputs[:, :-1], task=dual_task)

        # calculate reconstruction reward
        dual_scores = self.reward_model(logscore, rec_outputs, rec_out_lens, choice=task + '_rec').contiguous().view(-1, self.sample_size)
        rec_reward = dual_scores.detach()
        baseline = rec_reward.mean(dim=-1, keepdim=True)
        rec_reward = rec_reward - baseline

        if self.reward_model.reward_type == 'rel':
            total_reward = (1 - coefficient) * rec_reward
        elif 'rel' in self.reward_model.reward_type:
            total_reward = coefficient * val_reward + (1 - coefficient) * rec_reward
        else: total_reward = coefficient * val_reward
        primal_loss = - torch.mean(total_reward * primal_scores, dim=1)
        dual_loss = - torch.mean((1 - coefficient) * dual_scores, dim=1)
        return torch.sum(primal_loss), torch.sum(dual_loss)


    def construct_reverse_input(self, input_list, output_list, dual_task='nl2cf', device='cpu'):
        if dual_task == 'nl2cf':
            ex_list = [Example(nl=inp, cf=out) for inp, out in zip(input_list, output_list)]
        else: ex_list = [Example(nl=out, cf=inp) for inp, out in zip(input_list, output_list)]
        inputs, lens, outputs, out_lens = get_minibatch_paraphrase(ex_list, device, input_side=dual_task[:2], labeled=True)
        return inputs, lens, outputs, out_lens
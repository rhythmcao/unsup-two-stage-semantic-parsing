#coding=utf8
import os, sys
import torch
import torch.nn as nn
from utils.example import Example
from utils.constants import PAD, UNK, BOS, EOS
from utils.batch import get_minibatch_nl2cf, get_minibatch_cf2nl
from utils.domain.domain_base import Domain

class DualLearning(nn.Module):

    def __init__(self, nl2cf_model, cf2nl_model, reward_model, nl2cf_vocab, cf2nl_vocab, shared_encoder=False,
        alpha=0.5, beta=0.5, sample=5, reduction='sum', nl2cf_device=None, cf2nl_device=None, **kargs):
        """
            @args:
                1. alpha: reward for cycle starting from nl2cf_reward = val_reward * alpha + rec_reward * (1 - alpha)
                2. beta: reward for cycle starting from cf2nl_reward = val_reward * beta + rec_reward * (1 - beta)
                3. sample: beam search and sample size for training in dual learning cycles
        """
        super(DualLearning, self).__init__()
        self.nl2cf_device = nl2cf_device
        self.cf2nl_device = cf2nl_device
        self.nl2cf_model = nl2cf_model.to(self.nl2cf_device)
        self.cf2nl_model = cf2nl_model.to(self.cf2nl_device)
        if shared_encoder:
            self.cf2nl_model.src_embed = self.nl2cf_model.src_embed
            self.cf2nl_model.encoder = self.nl2cf_model.encoder
        self.shared_encoder = shared_encoder
        self.reward_model = reward_model
        self.alpha, self.beta, self.sample = alpha, beta, sample
        self.reduction = reduction
        self.nl2cf_vocab = nl2cf_vocab
        self.cf2nl_vocab = cf2nl_vocab

    def forward(self, *args, start_from='nl2cf', **kargs):
        """
            @args:
                *args(tensors): positional arguments for nl2cf or cf2nl
                start_from(enum): nl2cf or cf2nl
        """
        if start_from == 'nl2cf':
            return self.cycle_start_from_nl2cf(*args, **kargs)
        elif start_from == 'cf2nl':
            return self.cycle_start_from_cf2nl(*args, **kargs)
        else:
            raise ValueError('[Error]: dual learning cycle with unknown starting point !')

    def cycle_start_from_nl2cf(self, inputs, lens, raw_in):
        # primal model
        results = self.nl2cf_model.decode_batch(inputs, lens, self.nl2cf_vocab.cf2id, self.sample, self.sample)
        predictions, nl2cf_scores = results['predictions'], results['scores']
        predictions = [idx for each in predictions for idx in each]
        predictions = Domain().reverse(predictions, self.nl2cf_vocab.id2cf)
        raw_in = [each for each in raw_in for _ in range(self.sample)] # repeat sample times

        # calculate validity reward
        nl2cf_val_reward = self.reward_model(predictions, choice='nl2cf_val').contiguous().view(-1, self.sample)
        baseline = nl2cf_val_reward.mean(dim=-1, keepdim=True)
        nl2cf_val_reward -= baseline

        # dual model
        rev_inputs, rev_lens, rev_outputs, rev_out_lens = \
            self.nl2cf2nl(predictions, raw_in, vocab=self.cf2nl_vocab, device=self.cf2nl_device)
        logscore = self.cf2nl_model(rev_inputs, rev_lens, rev_outputs[:, :-1])

        # calculate reconstruction reward
        rec_reward = self.reward_model(logscore, rev_outputs, rev_out_lens, choice='rec').contiguous().view(-1, self.sample)
        nl2cf_rec_reward = torch.tensor(rec_reward.cpu().tolist(), dtype=torch.float, requires_grad=False)
        baseline = nl2cf_rec_reward.mean(dim=-1, keepdim=True)
        nl2cf_rec_reward -= baseline

        if self.reward_model.reward == 'rel':
            total_reward = (1 - self.alpha) * nl2cf_rec_reward
        elif 'rel' in self.reward_model.reward:
            total_reward = self.alpha * nl2cf_val_reward + (1 - self.alpha) * nl2cf_rec_reward
        else:
            total_reward = self.alpha * nl2cf_val_reward
        nl2cf_loss = - torch.mean(total_reward.to(self.nl2cf_device) * nl2cf_scores, dim=1)
        nl2cf_loss = torch.sum(nl2cf_loss) if self.reduction == 'sum' else torch.mean(nl2cf_loss)
        cf2nl_loss = - torch.mean((1 - self.alpha) * rec_reward, dim=1)
        cf2nl_loss = torch.sum(cf2nl_loss) if self.reduction == 'sum' else torch.mean(cf2nl_loss)
        return nl2cf_loss, cf2nl_loss

    def nl2cf2nl(self, cf_list, nl_list, vocab, device):
        ex_list = [ Example(['nl','cf'], cf=' '.join(cf), nl=' '.join(nl)) for cf, nl in zip(cf_list, nl_list) ]
        inputs, lens, outputs, out_lens, _ = get_minibatch_cf2nl(ex_list, vocab, device)
        return inputs, lens, outputs, out_lens

    def cycle_start_from_cf2nl(self, inputs, lens, raw_in):
        # primal model
        results = self.cf2nl_model.decode_batch(inputs, lens, self.cf2nl_vocab.nl2id, self.sample, self.sample)
        predictions, cf2nl_scores = results['predictions'], results['scores']
        predictions = [idx for each in predictions for idx in each]
        predictions = Domain().reverse(predictions, self.cf2nl_vocab.id2nl)
        raw_in = [each for each in raw_in for _ in range(self.sample)] # repeat sample times

        # calculate validity reward
        cf2nl_val_reward = self.reward_model(predictions, choice='cf2nl_val').contiguous().view(-1, self.sample)
        baseline = cf2nl_val_reward.mean(dim=-1, keepdim=True)
        cf2nl_val_reward -= baseline

        # dual model
        rev_inputs, rev_lens, rev_outputs, rev_out_lens = \
            self.cf2nl2cf(predictions, raw_in, self.nl2cf_vocab, self.nl2cf_device)
        logscore = self.nl2cf_model(rev_inputs, rev_lens, rev_outputs[:, :-1])

        # calculate reconstruction reward
        rec_reward = self.reward_model(logscore, rev_outputs, rev_out_lens, choice='rec').contiguous().view(-1, self.sample)
        cf2nl_rec_reward = torch.tensor(rec_reward.cpu().tolist(), dtype=torch.float, requires_grad=False)
        baseline = cf2nl_rec_reward.mean(dim=-1, keepdim=True)
        cf2nl_rec_reward -= baseline

        if self.reward_model.reward == 'rel':
            total_reward = (1 - self.beta) * cf2nl_rec_reward
        elif 'rel' in self.reward_model.reward:
            total_reward = self.beta * cf2nl_val_reward + (1 - self.beta) * cf2nl_rec_reward
        else:
            total_reward = self.beta * cf2nl_val_reward
        cf2nl_loss = - torch.mean(total_reward.to(self.cf2nl_device) * cf2nl_scores, dim=1)
        cf2nl_loss = torch.sum(cf2nl_loss) if self.reduction == 'sum' else torch.mean(cf2nl_loss)
        nl2cf_loss = - torch.mean((1 - self.beta) * rec_reward, dim=1)
        nl2cf_loss = torch.sum(nl2cf_loss) if self.reduction == 'sum' else torch.mean(nl2cf_loss)
        return nl2cf_loss, cf2nl_loss

    def cf2nl2cf(self, nl_list, cf_list, vocab, device):
        ex_list = [ Example(['nl','cf'], nl=' '.join(nl), cf=' '.join(cf)) for nl, cf in zip(nl_list, cf_list) ]
        inputs, lens, outputs, out_lens, _ = get_minibatch_nl2cf(ex_list, vocab, device)
        return inputs, lens, outputs, out_lens

    def decode_batch(self, *args, task='nl2cf', **kargs):
        if task == 'nl2cf':
            return self.nl2cf_model.decode_batch(*args, **kargs)
        elif task == 'cf2nl':
            return self.cf2nl_model.decode_batch(*args, **kargs)
        else:
            raise ValueError('[Error]: unknown task name !')

    def pad_embedding_grad_zero(self):
        self.nl2cf_model.pad_embedding_grad_zero()
        self.cf2nl_model.pad_embedding_grad_zero()

    def load_model(self, nl2cf_load_dir=None, cf2nl_load_dir=None):
        if nl2cf_load_dir is not None:
            self.nl2cf_model.load_model(nl2cf_load_dir)
        if cf2nl_load_dir is not None:
            self.cf2nl_model.load_model(cf2nl_load_dir)

    def save_model(self, nl2cf_save_dir=None, cf2nl_save_dir=None):
        if nl2cf_save_dir is not None:
            self.nl2cf_model.save_model(nl2cf_save_dir)
        if cf2nl_save_dir is not None:
            self.cf2nl_model.save_model(cf2nl_save_dir)

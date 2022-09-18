#coding=utf8
from utils.constants import *
from models.model_utils import lens2mask
import numpy as np
import torch
from utils.bleu import get_bleu_score
from utils.example import Example

class RewardModel():

    def __init__(self, nl_lm, cf_lm, sp_model, sty_model, lm_vocab, sp_vocab, sty_vocab, lm_device='cpu', sp_device='cpu', sty_device='cpu', reward="flu+sty+rel"):
        super(RewardModel, self).__init__()
        self.nl_lm, self.cf_lm = nl_lm.to(lm_device), cf_lm.to(lm_device)
        self.lm_vocab = lm_vocab
        self.lm_device = lm_device
        self.sp_model = sp_model.to(sp_device)
        self.sp_vocab = sp_vocab
        self.sp_device = sp_device
        self.sty_model = sty_model.to(sty_device)
        self.sty_vocab = sty_vocab
        self.sty_device = sty_device
        self.reward = reward

    def forward(self, *args, choice='nl2cf_val'):
        if choice == 'nl2cf_val':
            return self.nl2cf_validity_reward(*args)
        elif choice == 'cf2nl_val':
            return self.cf2nl_validity_reward(*args)
        elif choice == 'rec':
            return self.reconstruction_reward(*args)
        elif choice == 'nl2cf_rec':
            return self.nl2cf_reconstruction_reward(*args)
        elif choice == 'cf2nl_rec':
            return self.cf2nl_reconstruction_reward(*args)
        else:
            raise ValueError('[Error]: unknown reward choice !')

    def nl2cf_validity_reward(self, cf_list):
        val_reward = torch.zeros(len(cf_list), dtype=torch.float, device='cpu')
        if 'flu' in self.reward:
            # calculate canonical form language model length normalized log probability
            input_idxs = [[self.lm_vocab.cf2id[BOS]] + [self.lm_vocab.cf2id.get(word, self.lm_vocab.cf2id[UNK]) for word in sent] + [self.lm_vocab.cf2id[EOS]] for sent in cf_list]
            lens = [len(each) for each in input_idxs]
            max_len = max(lens)
            input_idxs = [sent + [self.lm_vocab.cf2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
            input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.lm_device)
            lens = torch.tensor(lens, dtype=torch.long, device=self.lm_device)
            self.cf_lm.eval()
            with torch.no_grad():
                logprob = self.cf_lm.sent_logprobability(input_tensor, lens).cpu()
            # calculate execution reward, 0/1 indicator
            # input_idxs = [[self.sp_vocab.cf2id.get(word, self.sp_vocab.cf2id[UNK]) for word in sent] for sent in cf_list]
            # lens = [len(each) for each in input_idxs]
            # max_len = max(lens)
            # input_idxs = [sent + [self.sp_vocab.cf2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
            # input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.sp_device)
            # lens_tensor = torch.tensor(lens, dtype=torch.long, device=self.sp_device)
            # self.sp_model.eval()
            # domain = Example.domain
            # with torch.no_grad():
                # results = self.sp_model.decode_batch(input_tensor, lens_tensor, self.sp_vocab.lf2id, beam_size=5, n_best=1)
                # predictions = results['predictions']
                # predictions = [each[0] for each in predictions]
                # predictions = domain.reverse(predictions, self.sp_vocab.id2lf)
                # ans = domain.obtain_denotations(domain.normalize(predictions))
                # grammar_check = domain.is_valid(ans)
                # grammar_check = torch.tensor(grammar_check, dtype=torch.float, device='cpu')
            # val_reward += logprob + grammar_check
            val_reward += logprob
        if 'sty' in self.reward:
            # calculate sty reward
            input_idxs = [[self.sty_vocab.nl2id.get(w, self.sty_vocab.nl2id[PAD]) for w in sent] for sent in cf_list]
            max_sent_len = max([len(sent) for sent in input_idxs])
            input_idxs = [sent + [self.sty_vocab.nl2id[PAD]] * (max_sent_len - len(sent)) for sent in input_idxs]
            input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.sty_device)
            with torch.no_grad():
                results = self.sty_model(input_tensor)
                dis = results.cpu().float()
            val_reward += dis
        return val_reward

    def cf2nl_validity_reward(self, nl_list):
        val_reward = torch.zeros(len(nl_list), dtype=torch.float, device='cpu')
        if 'flu' in self.reward:
            # calculate natural language model length normalized log probability
            input_idxs = [[self.lm_vocab.nl2id[BOS]] + [self.lm_vocab.nl2id.get(word, self.lm_vocab.nl2id[UNK]) for word in sent] + [self.lm_vocab.nl2id[EOS]] for sent in nl_list]
            lens = [len(each) for each in input_idxs]
            max_len = max(lens)
            input_idxs = [sent + [self.lm_vocab.nl2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
            input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.lm_device)
            lens = torch.tensor(lens, dtype=torch.long, device=self.lm_device)
            self.nl_lm.eval()
            with torch.no_grad():
                logprob = self.nl_lm.sent_logprobability(input_tensor, lens).cpu()
            val_reward += logprob
        if 'sty' in self.reward:
            # calculate sty reward
            input_idxs = [[self.sty_vocab.nl2id.get(w, self.sty_vocab.nl2id[PAD]) for w in sent] for sent in nl_list]
            max_sent_len = max([len(sent) for sent in input_idxs])
            input_idxs = [sent + [self.sty_vocab.nl2id[PAD]] * (max_sent_len - len(sent)) for sent in input_idxs]
            input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.sty_device)
            with torch.no_grad():
                results = self.sty_model(input_tensor)
                dis = results.cpu().float()
            val_reward += (1 - dis)
        return val_reward

    def reconstruction_reward(self, logscores, references, lens):
        """
            logscores: bsize x max_out_len - 1 x vocab_size
            references: bsize x max_out_len
            lens: len for each sample
        """
        references, lens = references[:, 1:], lens - 1
        mask = lens2mask(lens)
        pick_score = torch.gather(logscores, dim=-1, index=references.unsqueeze(dim=-1)).squeeze(dim=-1)
        masked_score = mask.float() * pick_score
        reward = masked_score.sum(dim=1)
        return reward

    def nl2cf_reconstruction_reward(self, predictions, references):
        references = [[ref] for ref in references]
        scores = []
        for pred, ref in zip(predictions, references):
            scores.append(get_bleu_score(pred, ref))
        return torch.tensor(scores, dtype=torch.float)

    def cf2nl_reconstruction_reward(self, predictions, references):
        references = [[ref] for ref in references]
        scores = []
        for pred, ref in zip(predictions, references):
            scores.append(get_bleu_score(pred, ref, weights=(0, 0, 0.5, 0.5)))
        return torch.tensor(scores, dtype=torch.float)

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)

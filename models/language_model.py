#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import rnn_wrapper, lens2mask


class DualLanguageModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(DualLanguageModel, self).__init__()
        self.nl_lm = LanguageModel(*args, **kwargs)
        self.cf_lm = LanguageModel(*args, **kwargs)
        # share word embeddings and decoder tied
        if kwargs.get('share_encoder', True):
            self.cf_lm.encoder = self.nl_lm.encoder
        self.nl_lm.decoder.weight = self.nl_lm.encoder.weight
        self.cf_lm.decoder.weight = self.cf_lm.encoder.weight


    def forward(self, inputs, lens, input_side='nl'):
        if input_side == 'nl':
            return self.nl_lm(inputs, lens)
        else:
            return self.cf_lm(inputs, lens)


    def sentence_logprob(self, inputs, lens, input_side='nl'):
        if input_side == 'nl':
            return self.nl_lm.sentence_logprob(inputs, lens)
        else:
            return self.cf_lm.sentence_logprob(inputs, lens)


class LanguageModel(nn.Module):
    """ Traditional RNN Language Model
    """
    def __init__(self, vocab_size=None, pad_idx=0, embed_size=100, hidden_size=200,
            num_layers=1, cell='lstm', dropout=0.5, init_weight=0.2, **kwargs):
        super(LanguageModel, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.cell = cell.upper() # RNN/LSTM/GRU
        self.rnn = getattr(nn, self.cell)(embed_size, hidden_size, num_layers, batch_first=True,
            bidirectional=False, dropout=(dropout if num_layers > 1 else 0))
        self.affine = nn.Linear(hidden_size, embed_size)
        self.decoder = nn.Linear(embed_size, vocab_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if init_weight is not None and init_weight > 0:
            for p in self.parameters():
                p.data.uniform_(-init_weight, init_weight)
            self.encoder.weight.data[pad_idx].zero_()


    def forward(self, inputs, lens):
        inputs, lens = inputs[:, :-1], lens - 1
        word_embeds = self.dropout_layer(self.encoder(inputs)) # bsize x seqlen x embed_size
        outputs = rnn_wrapper(self.rnn, word_embeds, lens, self.cell, need_hidden_states=False)
        hiddens = self.decoder(self.affine(self.dropout_layer(outputs)))
        scores = F.log_softmax(hiddens, dim=-1)
        return scores


    def sentence_logprob(self, inputs, lens):
        ''' Given sentences, calculate its length-normalized log-probability
        inputs must contain BOS and EOS symbol.
        @args:
            inputs: torch.FloatTensor, bsize x seqlen
            lens: torch.LongTensor, bsize
        @return:
            length-normalized logprob score, torch.FloatTensor, bsize
        '''
        lens = lens - 1
        inputs, targets = inputs[:, :-1], inputs[:, 1:]
        word_embeds = self.dropout_layer(self.encoder(inputs)) # bsize x seqlen x embed_size
        outputs = rnn_wrapper(self.rnn, word_embeds, lens, self.cell, need_hidden_states=False)
        hiddens = self.decoder(self.affine(self.dropout_layer(outputs)))
        scores = F.log_softmax(hiddens, dim=-1)
        logprobs = torch.gather(scores, 2, targets.unsqueeze(-1)).contiguous().view(outputs.size(0), outputs.size(1))
        sentence_logprobs = torch.sum(logprobs.masked_fill_(~lens2mask(lens), 0.), dim=-1)
        return sentence_logprobs / lens.float()
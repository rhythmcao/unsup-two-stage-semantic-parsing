#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


def tile(x, count, dim=0):
    """
        Tiles x on dimension dim count times.
        E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
            [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
        Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x


def lens2mask(lens):
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks


def rnn_wrapper(encoder, inputs, lens, cell='lstm', need_hidden_states=True):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, bsize x max_seq_len x in_dim
            lens(torch.LongTensor): seq len for each sample, bsize
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_size*2
            if need_hidden_states:
                hidden_states(tuple or torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_size
    """
    # rerank according to lens and temporarily remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_index = torch.sum(sorted_lens > 0).item()
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key[:nonzero_index])
    # forward non empty inputs    
    packed_inputs = rnn_utils.pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_index].tolist(), batch_first=True)
    packed_out, h = encoder(packed_inputs)  # bsize x srclen x dim
    out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    # pad zeros due to empty inputs
    pad_zeros = out.new_zeros(sorted_lens.size(0) - out.size(0), out.size(1), out.size(2))
    sorted_out = torch.cat([out, pad_zeros], dim=0)    
    # rerank according to sort_key
    shape = list(sorted_out.size())
    out = sorted_out.new_zeros(sorted_out.size()).scatter_(0, sort_key.unsqueeze(-1).unsqueeze(-1).expand(*shape), sorted_out)
    if not need_hidden_states: return out

    if cell.upper() == 'LSTM': h, c = h
    pad_hiddens = h.new_zeros(h.size(0), sorted_lens.size(0) - h.size(1), h.size(2))
    sorted_hiddens = torch.cat([h, pad_hiddens], dim=1)
    shape = list(sorted_hiddens.size())
    hiddens = sorted_hiddens.new_zeros(sorted_hiddens.size()).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).expand(*shape), sorted_hiddens)
    if cell.upper() == 'LSTM':
        pad_cells = c.new_zeros(c.size(0), sorted_lens.size(0) - c.size(1), c.size(2))
        sorted_cells = torch.cat([c, pad_cells], dim=1)
        shape = list(sorted_cells.size())
        cells = sorted_cells.new_zeros(sorted_cells.size()).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).expand(*shape), sorted_cells)
        return out, (hiddens.contiguous(), cells.contiguous())
    return out, hiddens.contiguous()


class RNNEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, padding_idx=0, dropout=0.5) -> None:
        super(RNNEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, *args):
        return self.dropout_layer(self.embed(args[0]))


class RNNEncoder(nn.Module):
    """ RNN encoder is a wrapper of the Pytorch LSTM/GRU layers
    """
    def __init__(self, input_size, hidden_size, num_layers, cell="lstm", bidirectional=True, dropout=0.5):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if self.num_layers > 1 else 0
        self.cell = cell.upper()
        self.rnn_encoder = getattr(nn, self.cell)(self.input_size, self.hidden_size, 
                        num_layers=self.num_layers, bidirectional=self.bidirectional, 
                        batch_first=True, dropout=self.dropout)
        
    def forward(self, x, lens, need_hidden_states=True):
        """ Pass the x and lens through each RNN layer.
        """
        return rnn_wrapper(self.rnn_encoder, x, lens, cell=self.cell, need_hidden_states=need_hidden_states)  # bsize x srclen x dim


class Attention(nn.Module):
    METHODS = ['general', 'feedforward']

    def __init__(self, enc_dim, dec_dim, method='feedforward'):
        super(Attention, self).__init__()
        self.enc_dim, self.dec_dim = enc_dim, dec_dim
        assert method in Attention.METHODS
        self.method = method
        if self.method == 'general':
            self.Wa = nn.Linear(self.enc_dim, self.dec_dim, bias=False)
        else:
            self.Wa = nn.Linear(self.enc_dim + self.dec_dim, self.dec_dim, bias=False)
            self.Va = nn.Linear(self.dec_dim, 1, bias=False)


    def forward(self, memory, decoder_state, mask=None):
        '''
        @args:
            memory : bsize x src_len x enc_dim
            decoder_state : bsize x tgt_len x dec_dim
            mask : bsize x src_lens, BoolTensor
        @return: 
            context : bsize x tgt_len x enc_dim
        '''
        if self.method == 'general':
            m = self.Wa(memory) # bsize x src_len x dec_dim
            e = torch.bmm(decoder_state, m.transpose(1, 2)) # bsize x tgt_len x src_len
        else:
            d = decoder_state.unsqueeze(-2).expand(-1, -1, memory.size(1), -1) # bsize x tgt_len x src_len x dec_dim
            m = memory.unsqueeze(1).expand(-1, d.size(1), -1, -1) # bsize x tgt_len x src_len x enc_dim
            e = self.Wa(torch.cat([d, m], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1) # bsize x tgt_len x src_len
        if mask is not None:
            e.masked_fill_(~ mask.unsqueeze(1), - float('inf'))
        a = torch.softmax(e, dim=-1) 
        context = torch.bmm(a, memory) # bsize x tgt_len x enc_dim
        return context


class RNNDecoder(nn.Module):
    """ Generic unidirectional RNN layers containing Attention modules.
    """
    def __init__(self, input_size, hidden_size, num_layers, attn_model, cell="lstm", dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0
        self.cell = cell.upper()
        self.rnn_decoder = getattr(nn, self.cell)(self.input_size, self.hidden_size,
            num_layers=self.num_layers, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.attn_model = attn_model
        self.affine = nn.Linear(self.hidden_size + self.attn_model.enc_dim, self.input_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hidden_states, memory, src_mask):
        """
        @args
            x: decoder input embeddings, bsize x tgt_len x input_size
            hidden_states: previous decoder state
            memory: encoder output, bsize x src_len x hidden_size*2
            src_mask: bsize x src_lens
        """
        out, hidden_states = self.rnn_decoder(x, hidden_states)
        context = self.attn_model(memory, out, src_mask)
        feats = torch.cat([out, context], dim=-1)
        feats = self.affine(self.dropout_layer(feats))
        return feats, hidden_states


class Generator(nn.Module):
    """ Define standard linear + softmax generation step.
    """
    def __init__(self, input_size, vocab_size, dropout=0.5):
        super(Generator, self).__init__()
        self.proj = nn.Linear(input_size, vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        return F.log_softmax(self.proj(self.dropout_layer(x)), dim=-1)


class PoolingFunction(nn.Module):

    """ Map a sequence of hidden_size dim vectors into one fixed size vector with dimension output_size """

    def __init__(self, hidden_size=256, output_size=256, bias=True, method='attentive-pooling'):
        super(PoolingFunction, self).__init__()
        assert method in ['first-pooling', 'mean-pooling', 'max-pooling', 'attentive-pooling']
        self.method = method
        if self.method == 'attentive-pooling':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=bias)
            )
        self.mapping_function = nn.Sequential(nn.Linear(hidden_size, output_size, bias=bias), nn.Tanh()) \
            if hidden_size != output_size else lambda x: x


    def forward(self, inputs, mask=None):
        """ @args:
                inputs(torch.FloatTensor): features, bs x max_len x hidden_size
                mask(torch.BoolTensor): mask for inputs, True denotes meaningful positions, False denotes padding positions, bs x max_len
            @return:
                outputs(torch.FloatTensor): aggregate seq_len dim for inputs, bs x output_size
        """
        if self.method == 'first-pooling':
            outputs = inputs[:, 0, :]
        elif self.method == 'max-pooling':
            outputs = inputs.masked_fill(~ mask.unsqueeze(-1), -1e8)
            outputs = outputs.max(dim=1)[0]
        elif self.method == 'mean-pooling':
            mask_float = mask.float().unsqueeze(-1)
            outputs = (inputs * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        elif self.method == 'attentive-pooling':
            e = self.attn(inputs).squeeze(-1)
            e = e + (1 - mask.float()) * (-1e20)
            a = torch.softmax(e, dim=1).unsqueeze(1)
            outputs = torch.bmm(a, inputs).squeeze(1)
        else:
            raise ValueError('[Error]: Unrecognized pooling method %s !' % (self.method))
        outputs = self.mapping_function(outputs)
        return outputs
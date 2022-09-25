#coding=utf8
import torch
import torch.nn as nn
from utils.constants import BOS, EOS, MAX_DECODE_LENGTH
from models.model_utils import tile, lens2mask, RNNEmbedding, RNNEncoder, RNNDecoder, Attention, Generator
from models.beam import Beam, GNMTGlobalScorer


def dual_paraphrase_model(**kwargs):
    # src_vocab_size == tgt_vocab_size and src_pad_idx == tgt_pad_idx
    nl2cf_model = ParaphraseModel(**kwargs)
    cf2nl_model = ParaphraseModel(**kwargs)
    return DualParaphraseModel(nl2cf_model, cf2nl_model, kwargs.get('share_encoder', True))


class DualParaphraseModel(nn.Module):

    def __init__(self, nl2cf_model, cf2nl_model, share_encoder=True):
        super(DualParaphraseModel, self).__init__()
        self.nl2cf_model, self.cf2nl_model = nl2cf_model, cf2nl_model
        if share_encoder:
            self.cf2nl_model.src_embed = self.nl2cf_model.src_embed
            self.nl2cf_model.tgt_embed = self.nl2cf_model.src_embed
            self.cf2nl_model.tgt_embed = self.nl2cf_model.src_embed
            self.cf2nl_model.encoder = self.nl2cf_model.encoder


    def forward(self, *args, **kwargs):
        task = kwargs.pop('task', 'nl2cf')
        if task == 'nl2cf':
            return self.nl2cf_model(*args, **kwargs)
        else: return self.cf2nl_model(*args, **kwargs)


    def decode_batch(self, *args, **kwargs):
        task = kwargs.pop('task', 'nl2cf')
        if task == 'nl2cf':
            return self.nl2cf_model.decode_batch(*args, **kwargs)
        else: return self.cf2nl_model.decode_batch(*args, **kwargs)


class ParaphraseModel(nn.Module):

    def __init__(self, src_vocab_size=None, tgt_vocab_size=None, src_pad_idx=0, tgt_pad_idx=0, embed_size=100, hidden_size=200, num_layers=1,
            cell='lstm', dropout=0.5, init_weight=None, **kwargs):
        super(ParaphraseModel, self).__init__()
        self.src_embed = RNNEmbedding(src_vocab_size, embed_size, padding_idx=src_pad_idx, dropout=dropout)
        self.encoder = RNNEncoder(embed_size, hidden_size, num_layers, cell, dropout=dropout)
        self.cell, self.num_layers, self.hidden_size = cell.upper(), num_layers, hidden_size
        self.tgt_embed = RNNEmbedding(tgt_vocab_size, embed_size, padding_idx=tgt_pad_idx, dropout=dropout)
        attn_model = Attention(hidden_size * 2, hidden_size, method='feedforward')
        self.decoder = RNNDecoder(embed_size, hidden_size, num_layers, attn_model, cell=cell, dropout=dropout)
        self.generator = Generator(embed_size, tgt_vocab_size, dropout)

        if init_weight is not None and init_weight > 0:
            for p in self.parameters():
                p.data.uniform_(- init_weight, init_weight)
            self.src_embed.embed.weight.data[src_pad_idx].zero_()
            self.tgt_embed.embed.weight.data[tgt_pad_idx].zero_()


    def forward(self, src_inputs, src_lens, tgt_inputs):
        """ Used during training time.
        """
        memory = self.encoder(self.src_embed(src_inputs, src_lens.device), src_lens, need_hidden_states=False)
        h0 = memory.new_zeros(self.num_layers, memory.size(0), self.hidden_size).contiguous()
        hidden_states = (h0, h0.new_zeros(h0.size()).contiguous()) if self.cell == 'LSTM' else h0
        dec_out, _ = self.decoder(self.tgt_embed(tgt_inputs), hidden_states, memory, lens2mask(src_lens))
        out = self.generator(dec_out)
        return out


    def decode_batch(self, src_inputs, src_lens, vocab, beam_size=5, n_best=1, alpha=0.6, length_pen='avg', penalty=.0):
        memory = self.encoder(self.src_embed(src_inputs, src_lens.device), src_lens, need_hidden_states=False)
        h0 = memory.new_zeros(self.num_layers, memory.size(0), self.hidden_size).contiguous()
        hidden_states = (h0, h0.new_zeros(h0.size()).contiguous()) if self.cell == 'LSTM' else h0
        if beam_size == 1:
            return self.decode_greed(hidden_states, memory, lens2mask(src_lens), vocab)
        else:
            return self.decode_beam_search(hidden_states, memory, lens2mask(src_lens), vocab,
                beam_size=beam_size, n_best=n_best, alpha=alpha, length_pen=length_pen, penalty=penalty)


    def decode_greed(self, hidden_states, memory, src_mask, vocab):
        """
        @args:
            hidden_states: hidden_states from encoder
            memory: encoder output, bsize x src_len x enc_dim
            src_mask: ByteTensor, bsize x max_src_len
            vocab: tgt word2idx dict containing BOS, EOS
        """
        results = {"scores":[], "predictions":[]}

        # first target token is BOS
        batches = memory.size(0)
        ys = torch.ones(batches, 1, dtype=torch.long, device=memory.device).fill_(vocab[BOS])
        # record whether each sample is finished
        all_done = torch.tensor([False] * batches, dtype=torch.bool, device=memory.device)
        scores = torch.zeros(batches, 1, dtype=torch.float, device=memory.device)
        predictions = [[] for _ in range(batches)]

        for i in range(MAX_DECODE_LENGTH):
            logprob, hidden_states = self.decode_one_step(ys, hidden_states, memory, src_mask)
            maxprob, ys = torch.max(logprob, dim=1, keepdim=True)
            for i in range(batches):
                if not all_done[i]:
                    scores[i] += maxprob[i]
                    predictions[i].append(ys[i])
            done = ys.squeeze(dim=1) == vocab[EOS]
            all_done |= done
            if all_done.all(): break
        results["predictions"], results["scores"] = [[torch.cat(pred).tolist()] for pred in predictions], scores
        return results


    def decode_one_step(self, ys, hidden_states, memory, src_mask):
        """
            ys: bsize x 1
        """
        dec_out, hidden_states = self.decoder(self.tgt_embed(ys), hidden_states, memory, src_mask)
        out = self.generator(dec_out)
        return out.squeeze(dim=1), hidden_states


    def decode_beam_search(self, hidden_states, memory, src_mask, vocab,
            beam_size=5, n_best=1, alpha=0.6, length_pen='avg', penalty=.0):
        """ Beam search decoding
        """
        results = {"scores":[], "predictions":[]}

        # Construct beams, we donot use stepwise coverage penalty nor ngrams block
        remaining_sents = memory.size(0)
        global_scorer = GNMTGlobalScorer(alpha, length_pen)
        beam = [ Beam(beam_size, vocab, global_scorer=global_scorer, penalty=penalty, device=memory.device)
                for _ in range(remaining_sents) ]

        # repeat beam_size times
        memory, src_mask = tile([memory, src_mask], beam_size, dim=0)
        hidden_states = tile(hidden_states, beam_size, dim=1)
        h_c = type(hidden_states) in [list, tuple]
        batch_idx = list(range(remaining_sents))

        for i in range(MAX_DECODE_LENGTH):
            # (a) construct beamsize * remaining_sents next words
            ys = torch.stack([b.get_current_state() for b in beam if not b.done()]).contiguous().view(-1,1)

            # (b) pass through the decoder network
            out, hidden_states = self.decode_one_step(ys, hidden_states, memory, src_mask)
            out = out.contiguous().view(remaining_sents, beam_size, -1)

            # (c) advance each beam
            active, select_indices_array = [], []
            # Loop over the remaining_batch number of beam
            for b in range(remaining_sents):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beam[idx].advance(out[b])
                if not beam[idx].done():
                    active.append((idx, b))
                select_indices_array.append(beam[idx].get_current_origin() + b * beam_size)

            # (d) update hidden_states history
            select_indices_array = torch.cat(select_indices_array, dim=0)
            if h_c:
                hidden_states = (hidden_states[0].index_select(1, select_indices_array), hidden_states[1].index_select(1, select_indices_array))
            else:
                hidden_states = hidden_states.index_select(1, select_indices_array)

            if not active:
                break

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=memory.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            def update_active(t):
                if t is None: return t
                t_reshape = t.contiguous().view(remaining_sents, beam_size, -1)
                new_size = list(t.size())
                new_size[0] = -1
                return t_reshape.index_select(0, active_idx).view(*new_size)

            if h_c:
                hidden_states = (
                    update_active(hidden_states[0].transpose(0, 1)).transpose(0, 1).contiguous(),
                    update_active(hidden_states[1].transpose(0, 1)).transpose(0, 1).contiguous()
                )
            else:
                hidden_states = update_active(hidden_states.transpose(0, 1)).transpose(0, 1).contiguous()
            memory = update_active(memory)
            src_mask = update_active(src_mask)
            remaining_sents = len(active)

        for b in beam:
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyps.append(b.get_hyp(times, k)) # hyp contains </s> but does not contain <s>
            results["predictions"].append(hyps) # batch list of variable_tgt_len
            results["scores"].append(torch.stack(scores)[:n_best]) # list of [n_best], torch.FloatTensor
        results["scores"] = torch.stack(results["scores"])
        return results

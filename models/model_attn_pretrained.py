#coding=utf8
from models.model_attn import AttnModel
from models.model_utils import lens2mask

class AttnPretrainedElmoModel(AttnModel):

    def pad_embedding_grad_zero(self):
        self.tgt_embed.pad_embedding_grad_zero()

class AttnPretrainedBertModel(AttnModel):

    def forward(self, src_inputs, src_lens, tgt_inputs):
        """
            Used during training time.
        """
        embeds, src_lens = self.src_embed(src_inputs)
        enc_out, hidden_states = self.encoder(embeds, src_lens)
        hidden_states = self.enc2dec(hidden_states)
        src_mask = lens2mask(src_lens)
        dec_out, _ = self.decoder(self.tgt_embed(tgt_inputs), hidden_states, enc_out, src_mask)
        out = self.generator(dec_out)
        return out

    def decode_batch(self, src_inputs, src_lens, vocab, beam_size=5, n_best=1, alpha=0.6, length_pen='avg'):
        embeds, scr_lens = self.src_embed(src_inputs)
        enc_out, hidden_states = self.encoder(embeds, src_lens)
        hidden_states = self.enc2dec(hidden_states)
        src_mask = lens2mask(src_lens)
        if beam_size == 1:
            return self.decode_greed(hidden_states, enc_out, src_mask, vocab)
        else:
            return self.decode_beam_search(hidden_states, enc_out, src_mask, vocab,
                beam_size=beam_size, n_best=n_best, alpha=alpha, length_pen=length_pen)

    def pad_embedding_grad_zero(self):
        self.tgt_embed.pad_embedding_grad_zero()
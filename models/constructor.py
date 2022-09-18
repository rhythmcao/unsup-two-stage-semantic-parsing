#coding=utf8
import torch
import torch.nn as nn
from utils.constants import *
from models.embedding.embedding_rnn import RNNEmbeddings
from models.embedding.models_utils_elmo_bert_xlnet_layer import PretrainedInputEmbeddings
from models.encoder.encoder_rnn import RNNEncoder
from models.enc2dec.state_transition import StateTransition
from models.attention.attention_rnn import Attention
from models.decoder.decoder_rnn import RNNDecoder
from models.generator.generator_naive import Generator
from models.model_attn import AttnModel
from models.model_attn_pretrained import AttnPretrainedElmoModel, AttnPretrainedBertModel

def construct_model(
    src_vocab=None, tgt_vocab=None, pad_src_idxs=[0], pad_tgt_idxs=[0],
    src_emb_size=100, tgt_emb_size=100, hidden_dim=200, num_layers=1, bidirectional=True,
    trans='empty', cell='lstm', dropout=0.5, init=None, **kargs
):
    """
        Construct Seq2Seq model with attention mechanism
    """
    num_directions = 2 if bidirectional else 1
    enc2dec_model = StateTransition(num_layers, cell=cell, bidirectional=bidirectional, hidden_dim=hidden_dim, method=trans)
    attn_model = Attention(hidden_dim * num_directions, hidden_dim)
    src_embeddings = RNNEmbeddings(src_emb_size, src_vocab, pad_token_idxs=pad_src_idxs, dropout=dropout)
    encoder = RNNEncoder(src_emb_size, hidden_dim, num_layers, cell=cell, bidirectional=bidirectional, dropout=dropout)
    tgt_embeddings = RNNEmbeddings(tgt_emb_size, tgt_vocab, pad_token_idxs=pad_tgt_idxs, dropout=dropout)
    decoder = RNNDecoder(tgt_emb_size, hidden_dim, num_layers, attn=attn_model, cell=cell, dropout=dropout)
    generator_model = Generator(tgt_emb_size, tgt_vocab, dropout=dropout)
    model = AttnModel(src_embeddings, encoder, tgt_embeddings, decoder, enc2dec_model, generator_model)

    if init:
        for p in model.parameters():
            p.data.uniform_(-init, init)
        for pad_token_idx in pad_src_idxs:
            model.src_embed.embed.weight.data[pad_token_idx].zero_()
        for pad_token_idx in pad_tgt_idxs:
            model.tgt_embed.embed.weight.data[pad_token_idx].zero_()
    return model

def construct_pretrained_embed_model(
    tgt_vocab=None, pad_tgt_idxs=[0], tgt_emb_size=100, hidden_dim=200, num_layers=1, bidirectional=True,
    trans='empty', cell='lstm', dropout=0.5, init=None, device=None, emb_type='elmo', **kargs
):
    num_directions = 2 if bidirectional else 1
    enc2dec_model = StateTransition(num_layers, cell=cell, bidirectional=bidirectional, hidden_dim=hidden_dim, method=trans)
    attn_model = Attention(hidden_dim * num_directions, hidden_dim)
    emb_type_dict = {"bert": 'tf', 'elmo': 'elmo'}
    src_embeddings = PretrainedInputEmbeddings(pretrained_model_type=emb_type_dict[emb_type], dropout=dropout, device=device)
    src_emb_size = {"elmo": 1024, "bert": 768}
    encoder = RNNEncoder(src_emb_size[emb_type], hidden_dim, num_layers, cell=cell, bidirectional=bidirectional, dropout=dropout)
    tgt_embeddings = RNNEmbeddings(tgt_emb_size, tgt_vocab, pad_token_idxs=pad_tgt_idxs, dropout=dropout)
    decoder = RNNDecoder(tgt_emb_size, hidden_dim, num_layers, attn=attn_model, cell=cell, dropout=dropout)
    generator_model = Generator(tgt_emb_size, tgt_vocab, dropout=dropout)
    if emb_type == 'elmo':
        MODEL = AttnPretrainedElmoModel
    else:
        MODEL = AttnPretrainedBertModel
    model = MODEL(src_embeddings, encoder, tgt_embeddings, decoder, enc2dec_model, generator_model)

    if emb_type == 'bert':
        for p in model.src_embed.parameters():
            p.requires_grad = False

    if init:
        for p in model.encoder.parameters():
            p.data.uniform_(-init, init)
        for p in model.enc2dec.parameters():
            p.data.uniform_(-init, init)
        for p in model.tgt_embed.parameters():
            p.data.uniform_(-init, init)
        for p in model.decoder.parameters():
            p.data.uniform_(-init, init)
        for p in model.generator.parameters():
            p.data.uniform_(-init, init)
        for pad_token_idx in pad_tgt_idxs:
            model.tgt_embed.embed.weight.data[pad_token_idx].zero_()
    return model

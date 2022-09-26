#coding=utf8
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from allennlp.modules.elmo import Elmo, batch_to_ids
from models.model_utils import PoolingFunction, lens2mask
from models.paraphrase_model import ParaphraseModel


def semantic_parsing_model(**kwargs):
    embed_type = kwargs.get('pretrained_embed', 'glove')
    if embed_type == 'glove':
        return ParaphraseModel(**kwargs)
    elif embed_type == 'elmo':
        kwargs['embed_size'] = 1024
    else: # bert-base-uncased model
        kwargs['embed_size'] = 768
    return PretrainedSemanticParser(**kwargs)


def multitask_semantic_parsing_model(**kwargs):
    sp_model = ParaphraseModel(**kwargs)
    kwargs['tgt_vocab_size'], kwargs['tgt_pad_idx'] = kwargs['src_vocab_size'], kwargs['src_pad_idx']
    dae_model = ParaphraseModel(**kwargs)
    return MultitaskSemanticParser(sp_model, dae_model)


class MultitaskSemanticParser(nn.Module):

    def __init__(self, sp_model, dae_model):
        super(MultitaskSemanticParser, self).__init__()
        self.sp_model, self.dae_model = sp_model, dae_model
        self.dae_model.src_embed = self.sp_model.src_embed
        self.dae_model.tgt_embed = self.sp_model.src_embed
        self.dae_model.encoder = self.sp_model.encoder


    def forward(self, *args, **kwargs):
        task = kwargs.pop('task', 'semantic_parsing')
        if task == 'semantic_parsing':
            return self.sp_model(*args, **kwargs)
        else: return self.dae_model(*args, **kwargs)


    def decode_batch(self, *args, **kwargs):
        task = kwargs.pop('task', 'semantic_parsing')
        if task == 'semantic_parsing':
            return self.sp_model.decode_batch(*args, **kwargs)
        else: return self.dae_model.decode_batch(*args, **kwargs)


class PretrainedSemanticParser(ParaphraseModel):

    def __init__(self, **kwargs) -> None:
        super(PretrainedSemanticParser, self).__init__(**kwargs)
        embed_type = kwargs.get('pretrained_embed', 'bert')
        dropout = kwargs.get('dropout', 0.)
        if embed_type == 'bert': # replace the original module self.src_embed
            model_path = 'pretrained_models/bert-base-uncased'
            self.src_embed = BERTEmbedding(model_path, dropout=dropout)
        else: # elmo
            config_path = 'pretrained_models/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            model_path = 'pretrained_models/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
            self.src_embed = ELMoEmbedding(config_path, model_path, dropout=dropout)


class BERTEmbedding(nn.Module):

    def __init__(self, model_path, dropout=0.5, pooling_method='first-pooling') -> None:
        super(BERTEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.plm = BertModel.from_pretrained(model_path)
        self.pooling_function = PoolingFunction(self.plm.config.hidden_size, self.plm.config.hidden_size, method=pooling_method)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, inputs, device='cpu'):
        """
        @args:
            inputs: raw tokens (not padded), e.g. [ ['hello', 'world'] , ['i', 'love', 'nlp', '.'], ...]
            lens: torch.LongTensor, word length of each sample
        @return:
            embeddings: torch.FloatTensor, bs x max_len x embed_size(768), max_len is the raw input tokens instead of the subwords
        """
        lens = torch.tensor([len(ex) for ex in inputs], dtype=torch.long, device=device)
        max_len, mask = lens.max().item(), lens2mask(lens)
        tokens = [[self.tokenizer.tokenize(w) for w in ex] for ex in inputs]
        token_lens = torch.tensor([len(toks) for ex in tokens for toks in ex], dtype=torch.long, device=device)
        max_token_len, token_mask = token_lens.max().item(), lens2mask(token_lens)

        input_dict = self.tokenizer(inputs, is_split_into_words=True, padding=True, return_tensors='pt').to(device)
        outputs = self.plm(**input_dict)[0]

        # remove [CLS]/[SEP]/[PAD] embeddings and re-allocate the shape, token_num x max_token_len x embed_size
        remove_special_mask = input_dict['attention_mask'].bool()
        special_index = token_lens.new_zeros(remove_special_mask.size(0), 2)
        special_index[:, 1] = input_dict['attention_mask'].sum(dim=1) - 1
        remove_special_mask = remove_special_mask.scatter_(1, special_index, False)
        source = outputs.masked_select(remove_special_mask.unsqueeze(-1))
        token_embeddings = outputs.new_zeros(token_lens.size(0), max_token_len, outputs.size(-1)).masked_scatter_(token_mask.unsqueeze(-1), source)

        # subwords/word pieces pooling
        token_embeddings = self.pooling_function(token_embeddings, token_mask) # token_num x embed_size
        out_embeddings = outputs.new_zeros(len(inputs), max_len, outputs.size(-1)).masked_scatter_(mask.unsqueeze(-1), token_embeddings)
        return self.dropout_layer(out_embeddings)


class ELMoEmbedding(nn.Module):

    def __init__(self, config_path, model_path, dropout=0.5) -> None:
        super(ELMoEmbedding, self).__init__()
        self.plm = Elmo(config_path, model_path, 1, dropout=dropout)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, inputs, device='cpu'):
        """
        @args:
            inputs: raw tokens, e.g. [ ['hello', 'world'] , ['i', 'love', 'nlp', '.'], ...]
            lens: torch.LongTensor, word length of each sample, not used here
        @return:
            embeddings: torch.FloatTensor, bs x max_len x embed_size(1024)
        """
        tokens = batch_to_ids(inputs).to(device)
        embeddings = self.plm(tokens)['elmo_representations'][0]
        return self.dropout_layer(embeddings)

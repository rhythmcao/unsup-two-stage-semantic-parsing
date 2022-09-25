#coding=utf8
""" Word2vec utilities: load pre-trained embeddings of GloVe6B
"""
import os, torch
import numpy as np
from utils.constants import BOS, EOS, PAD, UNK


class Word2Vec():

    def __init__(self, embed_size, vocab) -> None:
        self.embed_size, self.word2vec = embed_size, {}
        if embed_size in [50, 100, 200, 300]:
            words = set(vocab.nl2id.keys())
            filename = os.path.join('pretrained_models', f'glove.6B.{embed_size}d.txt')
            self._read_word_embeddings(filename, words)


    def _read_word_embeddings(self, filename, vocab):
        mapping = {}
        mapping['bos'], mapping['eos'], mapping['padding'], mapping['unknown'] = BOS, EOS, PAD, UNK
        with open(filename, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                word = line[:line.index(' ')]
                word = mapping[word] if word in mapping else word
                if word in vocab and word not in self.word2vec:
                    values = line[line.index(' ') + 1:]
                    self.word2vec[word] = torch.from_numpy(np.fromstring(values, sep=' ', dtype=np.float))
        return self.word2vec


    def load_embeddings(self, module, word2id, device='cpu'):
        embed_size, cnt = module.weight.data.size(-1), 0
        assert embed_size == self.embed_size
        if embed_size not in [50, 100, 200, 300]:
            print('Donot use pretrained GloVe6B embeddings ...')
            return 0.0
        for word in word2id:
            if word in self.word2vec:
                module.weight.data[word2id[word]] = self.word2vec[word].to(device)
                cnt += 1
        return cnt / float(len(word2id))
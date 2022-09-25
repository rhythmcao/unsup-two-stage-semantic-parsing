#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import BOS, EOS, PAD, UNK

class Vocab():

    def __init__(self, dataset):
        super(Vocab, self).__init__()
        self.dataset = dataset
        dirname = os.path.join('data', self.dataset) if self.dataset == 'geo' else os.path.join('data', 'overnight')
        nl_path = os.path.join(dirname, dataset + '_vocab.nl')
        cf_path = os.path.join(dirname, dataset + '_vocab.cf')
        lf_path = os.path.join(dirname, dataset + '_vocab.lf')
        self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path)
        self.lf2id, self.id2lf = self.read_vocab(lf_path)

    def read_vocab(self, *args, bos_eos=True, pad=True, unk=True):
        word2idx, idx2word = {}, []
        if pad:
            word2idx[PAD] = len(word2idx)
            idx2word.append(PAD)
        if unk:
            word2idx[UNK] = len(word2idx)
            idx2word.append(UNK)
        if bos_eos:
            word2idx[BOS] = len(word2idx)
            idx2word.append(BOS)
            word2idx[EOS] = len(word2idx)
            idx2word.append(EOS)
        for vocab_path in args:
            with open(vocab_path, 'r') as f:
                for line in f:
                    word = line.strip()
                    if word == '': continue
                    idx = len(word2idx)
                    if word not in word2idx:
                        word2idx[word] = idx
                        idx2word.append(word)
        return word2idx, idx2word
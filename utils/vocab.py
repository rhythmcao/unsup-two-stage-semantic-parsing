#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import BOS, EOS, PAD, UNK

class Vocab():

    def __init__(self, dataset, task):
        super(Vocab, self).__init__()
        self.dataset = dataset
        dirname = os.path.join('data', 'overnight') if self.dataset not in ['geo', 'scholar'] else \
            os.path.join('data', self.dataset)
        nl_path = os.path.join(dirname, dataset + '_vocab.nl')
        cf_path = os.path.join(dirname, dataset + '_vocab.cf')
        lf_path = os.path.join(dirname, dataset + '_vocab.lf')
        # by default, nl_vocab contains cf_vocab
        if task == 'nl2cf':
            self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path, bos_eos=False)
            self.cf2id, self.id2cf = self.read_vocab(cf_path, bos_eos=True)
        elif task == 'cf2nl':
            # add nl_path due to shared encoder tasks
            self.cf2id, self.id2cf = self.read_vocab(nl_path, cf_path, bos_eos=False)
            self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path, bos_eos=True)
        elif task == 'language_model':
            self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path, bos_eos=True)
            self.cf2id, self.id2cf = self.read_vocab(cf_path, bos_eos=True)
        elif task == 'semantic_parsing':
            self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path, bos_eos=False)
            self.cf2id, self.id2cf = self.read_vocab(cf_path, bos_eos=False)
            self.lf2id, self.id2lf = self.read_vocab(lf_path, bos_eos=True)
        elif task == 'multi_task':
            self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path, bos_eos=False)
            self.cf2id, self.id2cf = self.read_vocab(nl_path, cf_path, bos_eos=True)
            self.lf2id, self.id2lf = self.read_vocab(lf_path, bos_eos=True)
        elif task == 'discriminator':
            self.nl2id, self.id2nl = self.read_vocab(nl_path, cf_path, bos_eos=False)
            self.cf2id, self.id2cf = self.read_vocab(nl_path, cf_path, bos_eos=False)
        else:
            raise ValueError('Unknown task name !')

    def read_vocab(self, *args, bos_eos=True, pad=True, unk=True, separator=' : '):
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
                    line = line.strip()
                    if line == '':
                        continue
                    if separator in line:
                        word, _ = line.split(separator)
                    else:
                        word = line
                    idx = len(word2idx)
                    if word not in word2idx:
                        word2idx[word] = idx
                        idx2word.append(word)
        return word2idx, idx2word

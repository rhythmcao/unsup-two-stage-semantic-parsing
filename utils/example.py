#coding=utf8
import numpy as np
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.domain import Domain
from utils.vocab import Vocab
from utils.word2vec import Word2Vec
from utils.noisy_channel import NoisyChannel


def split_dataset(dataset, split_ratio=1.0, seed=999):
    assert split_ratio >= 0. and split_ratio <= 1.0
    index = np.arange(len(dataset))
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    np.random.shuffle(index)
    splt = int(len(dataset) * split_ratio)
    first = [dataset[idx] for idx in index[:splt]]
    second = [dataset[idx] for idx in index[splt:]]
    if seed is not None:
        np.random.set_state(st0)
    return first, second


class Example():

    @classmethod
    def configuration(cls, dataset, embed_size=None, noise_type='none'):
        cls.dataset = dataset # dataset name
        cls.vocab = Vocab(cls.dataset)
        cls.noise = NoisyChannel(cls.dataset, noise_type) # class Noise object
        cls.domain = Domain(dataset) # class Domain object
        if cls.dataset == 'geo': dataset_path, data_splits = os.path.join('data', cls.dataset), ['train', 'dev', 'test']
        else: dataset_path, data_splits = os.path.join('data', 'overnight'), ['train', 'test']
        cls.file_paths = [os.path.join(dataset_path, f'{cls.dataset}_{split}.tsv') for split in data_splits]
        cls.word2vec = Word2Vec(embed_size, cls.vocab) if embed_size is not None else None


    def __init__(self, nl=[], cf=[], lf=[], variables={}):
        super(Example, self).__init__()
        self.nl = [w for w in nl.split(' ') if w != ''] if type(nl) == str else nl
        self.cf = [w for w in cf.split(' ') if w != ''] if type(cf) == str else cf
        self.lf = [w for w in lf.split(' ') if w != ''] if type(lf) == str else lf
        self.variables = variables


    @classmethod
    def load_dataset(cls, choice='train'):
        """ return example list of train, test or extra
        """
        if choice == 'train':
            train_dataset = cls.load_dataset_from_file(cls.file_paths[0])
            if len(cls.file_paths) == 3:
                dev_dataset = cls.load_dataset_from_file(cls.file_paths[1])
            else: # OVERNIGHT datasets have no official dev dataset, split train dataset
                train_dataset, dev_dataset = split_dataset(train_dataset, split_ratio=0.8, seed=999)
            return train_dataset, dev_dataset
        elif choice == 'test':
            test_dataset = cls.load_dataset_from_file(cls.file_paths[-1])
            return test_dataset
        else: raise ValueError(f'Fail to load dataset, unknown data split {choice} ...')

    @classmethod
    def load_dataset_from_file(cls, path):
        ex_list = []
        with open(path, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                line = line.split('\t')
                if len(line) == 3: (nl, cf, lf), v = line, {}
                else:
                    nl, cf, lf, v = line
                    v = json.loads(v.strip())
                ex_list.append(Example(nl=nl, cf=cf, lf=lf, variables=v))
        return ex_list


    @classmethod
    def create_faked_dataset_based_on_wmd(cls, dataset):
        cfs = list(map(lambda ex: ex.cf, dataset))
        most_similar_idxs = [cls.noise.pick_candidate(ex.nl, cfs)[1] for ex in dataset]
        faked_samples = [cls(nl=ex.nl, cf=dataset[idx].cf, lf=dataset[idx].lf, variables=dataset[idx].variables) for ex, idx in zip(dataset, most_similar_idxs)]
        return faked_samples


class UtteranceExample(Example):

    def __init__(self, nl=[], label=0):
        self.nl = nl.split(' ') if type(nl) == str else nl
        self.label = int(label)

    @classmethod
    def from_dataset(cls, dataset):
        return_dataset = []
        for ex in dataset:
            return_dataset.append(cls(nl=ex.nl, label=0))
            return_dataset.append(cls(nl=ex.cf, label=1))
        return return_dataset
#coding=utf8
import numpy as np
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.domain.domain_base import Domain
from utils.constants import DOMAINS
from utils.noise import Noise

def split_dataset(dataset, split_ratio=1.0, fixed=None):
    assert split_ratio >= 0. and split_ratio <= 1.0
    index = np.arange(len(dataset))
    if fixed:
        st0 = np.random.get_state()
        np.random.seed(fixed)
    np.random.shuffle(index)
    splt = int(len(dataset) * split_ratio)
    first = [dataset[idx] for idx in index[:splt]]
    second = [dataset[idx] for idx in index[splt:]]
    if fixed:
        np.random.set_state(st0)
    return first, second

def check_overlapping(dataset1, dataset2):
    cfs1 = [' '.join(ex.cf) for ex in dataset1]
    cfs2 = [' '.join(ex.cf) for ex in dataset2]
    set1, set2 = set(cfs1), set(cfs2)
    print('Distinct canonical forms in first/second dataset is: %d/%d' % (len(set1), len(set2)))
    print('Intersection of canonical forms in these two datasets is: %d' % (len(set1 & set2)))

def split_dataset_according_to_cf(dataset, split_ratio=0.5):
    assert split_ratio >= 0. and split_ratio <= 1.0
    index = np.arange(len(dataset))
    splt = int(len(dataset) * split_ratio)
    cfs = [' '.join(ex.cf) for ex in dataset]
    set_cfs = sorted(set(cfs), key=cfs.index)
    cf_index = np.arange(len(set_cfs))
    np.random.shuffle(cf_index)
    count_cfs = [cfs.count(set_cfs[idx]) for idx in cf_index]
    first, second, idx, count = [], [], 0, 0
    while count < splt:
        count += count_cfs[idx]
        idx += 1
    first_set, second_set = [set_cfs[i] for i in cf_index[:idx]], [set_cfs[i] for i in cf_index[idx:]]
    first = [ex for ex in dataset if ' '.join(ex.cf) in first_set]
    second = [ex for ex in dataset if ' '.join(ex.cf) in second_set]
    return first, second

class Example():

    @classmethod
    def set_domain(cls, dataset, drop=False, add=False, shuffle=False):
        assert dataset in DOMAINS
        cls.dataset = dataset # dataset name
        cls.noise = Noise(cls.dataset, drop, add, shuffle) # class Noise object
        cls.domain = Domain.from_dataset(dataset) # class Domain object
        if cls.dataset in ['geo', 'scholar']:
            cls.file_paths = [
                os.path.join('data', dataset, dataset + '_train.tsv'),
                os.path.join('data', dataset, dataset + '_dev.tsv'),
                os.path.join('data', dataset, dataset + '_test.tsv')
            ]
        else:
            cls.file_paths = [
                os.path.join('data', 'overnight', dataset + '_train.tsv'),
                os.path.join('data', 'overnight', dataset + '_test.tsv')
            ]

    def __init__(self, need, nl='', cf='', lf='', conf=1.0, variables=None):
        super(Example, self).__init__()
        for n in need:
            assert n in ['nl', 'cf', 'lf']
            sent = eval(n).split(' ')
            exec('self.' + n + " = [each for each in sent if each != '']")
        self.conf = conf
        self.variables = variables

    @classmethod
    def load_dataset(cls, choice='train'):
        """
            return example list of train, test or extra
        """
        if choice == 'train':
            train_dataset = cls.load_dataset_from_file(cls.file_paths[0])
            if len(cls.file_paths) == 3:
                dev_dataset = cls.load_dataset_from_file(cls.file_paths[1])
            else:
                # no dev dataset, split train dataset
                train_dataset, dev_dataset = split_dataset(train_dataset, split_ratio=0.8, fixed=999)
            return train_dataset, dev_dataset
        elif choice == 'test':
            test_dataset = cls.load_dataset_from_file(cls.file_paths[-1])
            return test_dataset
        else:
            raise ValueError('Fail to load dataset ...')

    @classmethod
    def load_dataset_from_file(cls, path):
        ex_list = []
        with open(path, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                if len(line.split('\t')) == 3:
                    nl, cf, lf = line.split('\t')
                    v = {}
                else:
                    nl, cf, lf, v = line.split('\t')
                    v = json.loads(v.strip())
                ex_list.append(Example(['nl', 'cf', 'lf'], nl=nl, cf=cf, lf=lf, variables=v))
        return ex_list

class UtteranceExample(Example):

    def __init__(self, nl='', label=0):
        self.nl = nl.split(' ')
        self.label = int(label)

    @classmethod
    def from_dataset(cls, dataset):
        return_dataset = []
        for ex in dataset:
            return_dataset.append(cls(nl=' '.join(ex.nl), label=0))
            return_dataset.append(cls(nl=' '.join(ex.cf), label=1))
        return return_dataset

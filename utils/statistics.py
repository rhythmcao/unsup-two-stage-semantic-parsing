#coding=utf8
"""
    Construct vocabulary for each dataset.
"""
import os, sys, argparse, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import BOS, EOS, PAD, UNK, DOMAINS
import operator
from collections import Counter

def read_examples(path):
    ex_list = []
    with open(path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line == '':
                continue
            nl, cf, lf = line.split('\t')[:3]
            nl = [each.strip() for each in nl.strip().split(' ') if each.strip() != '']
            cf = [each.strip() for each in cf.strip().split(' ') if each.strip() != '']
            lf = [each.strip() for each in lf.strip().split(' ') if each.strip() != '']
            ex_list.append((nl, cf, lf))
    return ex_list

def save_vocab(idx2word, vocab_path):
    with open(vocab_path, 'w') as f:
        for idx in range(len(idx2word)):
            f.write(idx2word[idx] + '\n')

def construct_vocab(input_seqs, mwf=1):
    '''
        Construct vocabulary given input_seqs
        @params:
            1. input_seqs: e.g.
                [ ['what', 'flight'] , ['which', 'flight'] ]
                or
                [ 'what', 'flight', 'which', 'flight' ]
            2. mwf: minimum word frequency
        @return:
            1. word2idx(dict)
            2. idx2word(dict)
    '''
    vocab, word2idx, idx2word = {}, {}, []
    for seq in input_seqs:
        if type(seq) in [tuple, list]:
            for word in seq:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        else:
            if seq not in vocab:
                vocab[seq] = 1
            else:
                vocab[seq] += 1
    
    # Discard those special tokens if already exist
    if PAD in vocab: del vocab[PAD]
    if UNK in vocab: del vocab[UNK]
    if BOS in vocab: del vocab[BOS]
    if EOS in vocab: del vocab[EOS]

    sorted_words = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_words if x[1] >= mwf]
    for word in sorted_words:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word.append(word)
    return word2idx, idx2word

def word_count(input_seqs, count_path):
    all_words = sum(input_seqs, [])
    counter = Counter(all_words)
    json.dump(dict(counter), open(count_path, 'w'), indent=4)

def main(args=sys.argv[1:]):
    """
        Construct vocabulary for each dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', help='dataset name')
    parser.add_argument('--word_count', action='store_true', help='add word count')
    parser.add_argument('--mwf', type=int, default=1, help='minimum word frequency, if less than this int, not included')
    opt = parser.parse_args(args)
    all_dataset = [opt.dataset] if opt.dataset in DOMAINS else DOMAINS
    dirname = os.path.join('data', 'scholar') if opt.dataset == 'scholar' else \
        os.path.join('data', 'geo') if opt.dataset == 'geo' else os.path.join('data', 'overnight')
    for dataset in all_dataset:
        file_path = os.path.join(dirname, dataset + '_train.tsv') # not include test_file
        ex_list = read_examples(file_path)
        dev_file_path = os.path.join(dirname, dataset + '_dev.tsv')
        if os.path.exists(dev_file_path):
            ex_list += read_examples(dev_file_path)
        nl, cf, lf = list(zip(*ex_list))

        if opt.word_count:
            nl_count_path, cf_count_path = os.path.join(dirname, dataset + '_count.nl'), os.path.join(dirname, dataset + '_count.cf')
            word_count(nl, nl_count_path)
            word_count(cf, cf_count_path)

        nl_vocab_path, cf_vocab_path, lf_vocab_path = os.path.join(dirname, dataset + '_vocab.nl'), \
            os.path.join(dirname, dataset + '_vocab.cf'), os.path.join(dirname, dataset + '_vocab.lf')
        _, id2nl = construct_vocab(nl, mwf=opt.mwf)
        _, id2lf = construct_vocab(lf, mwf=opt.mwf)
        _, id2cf = construct_vocab(cf, mwf=opt.mwf)
        save_vocab(id2nl, nl_vocab_path)
        save_vocab(id2cf, cf_vocab_path)
        save_vocab(id2lf, lf_vocab_path)

if __name__ == '__main__':

    main()

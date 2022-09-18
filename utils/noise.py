#coding=utf8
import sys, os, json
import random, math
import numpy as np
from nltk.corpus import stopwords
import gensim

class Noise():

    def __init__(self, dataset, drop=False, add=False, shuffle=False):
        super(Noise, self).__init__()
        self.dataset = dataset
        if self.dataset in ['geo', 'scholar']:
            self.nl_counter = json.load(open('data/' + self.dataset + '/' + self.dataset + '_count.nl'))
            self.cf_counter = json.load(open('data/' + self.dataset + '/' + self.dataset + '_count.cf'))
            # geo_not_drop = ['_state_', '_city_', '_river_', '_place_']
            # scholar_not_drop = ['authorname0', 'authorname1', 'keyphrasename0', 'keyphrasename1', 'venuename0', 'venuename1', 'year0', 'misc0', 'journalname0', 'datasetname0', 'datasetname1', 'title0']
        else:
            self.nl_counter = json.load(open('data/overnight/' + self.dataset + '_count.nl'))
            self.cf_counter = json.load(open('data/overnight/' + self.dataset + '_count.cf'))
        self.drop, self.add, self.shuffle = drop, add, shuffle
        if self.add:
            self.model = gensim.models.KeyedVectors.load_word2vec_format('data/.cache/GoogleNews-vectors-negative300.bin.gz', binary=True)
            self.model.init_sims(replace=True)
            self.stop_words = stopwords.words('english')

    def noisy_channel(self, inputs, side='cf', train_dataset=None):
        """
            inputs: raw word list
            Typically we apply three operations in sequential:
                1. importance-aware drop: drop word with probability according to counter
                2. mix-source add: add words from chosen candidates
                3. n-gram shuffle: shuffle the whole sentence
        """
        outputs = inputs
        if self.drop:
            outputs = self.drop_noise(outputs, side=side, max_drop_rate=0.2)
        if self.add:
            picked_sentences = np.random.choice(np.arange(len(train_dataset)), 50, replace=False)
            picked_sentences = [train_dataset[idx].cf if side == 'nl' else train_dataset[idx].nl for idx in picked_sentences]
            candidate, _ = self.pick_candidate(inputs, picked_sentences)
            outputs = self.add_noise(outputs, candidate, n_gram=1, min_ratio=0.1, max_ratio=0.2)
        if self.shuffle:
            outputs = self.shuffle_noise(outputs, degree=3, n_gram=2)
        return outputs

    def drop_noise(self, inputs, side='nl', max_drop_rate=0.2):
        if side == 'nl':
            count = [self.nl_counter[w] if w in self.nl_counter else 0 for w in inputs]
        else:
            count = [self.cf_counter[w] if w in self.cf_counter else 0 for w in inputs]
        freq = np.array([each / float(sum(count)) for each in count])
        clipped_freq = np.clip(freq, 0., max_drop_rate)
        drop_prob = np.random.rand(len(clipped_freq))
        outputs = [w for idx, w in enumerate(inputs) if drop_prob[idx] > clipped_freq[idx]]
        if len(outputs) < 0.5 * len(inputs): # avoid dropping too much
            return inputs
        return outputs

    def pick_candidate(self, inputs, candidates):

        def sentence_sim_score(s1, s2):
            s1 = [w.lower() for w in s1 if w.lower() not in self.stop_words]
            s2 = [w.lower() for w in s2 if w.lower() not in self.stop_words]
            return self.model.wmdistance(s1, s2)

        scores = [sentence_sim_score(inputs, each) for each in candidates]
        return candidates[scores.index(min(scores))], scores.index(min(scores))
        # index = np.random.choice(np.arange(len(candidates)))
        # return candidates[index], index

    def add_noise(self, inputs, candidate, n_gram=1, min_ratio=0.1, max_ratio=0.2):
        """
            select some (min_ratio <= ratio <= max_ratio) words in candidate and
            insert these words to some position (insert_idx) in inputs
        """
        max_id = int((len(candidate) + n_gram - 1) / float(n_gram))
        candidate_ids = np.arange(max_id).repeat(n_gram)[:len(candidate)]
        add_length = np.random.randint(math.floor(max_id * min_ratio), math.floor(max_id * max_ratio) + 1)
        sample_ids = np.random.choice(np.arange(max_id), add_length, replace=False)
        sample_words = [candidate[idx] for idx in range(len(candidate)) if candidate_ids[idx] in sample_ids]
        insert_idx = np.random.randint(0, len(inputs) + 1)
        return inputs[:insert_idx] + sample_words + inputs[insert_idx:]

    def shuffle_noise(self, inputs, degree=None, n_gram=1):
        """
            degree: controls how slightly we shuffle, =1 will not shuffle
            n_gram: controls length of span we shuffle
        """
        if degree is None:
            degree = len(inputs)
        max_word_id = int((len(inputs) + n_gram - 1) / n_gram)
        word_ids = np.arange(max_word_id).repeat(n_gram)[:len(inputs)]
        noise = np.random.uniform(0, degree, len(inputs))
        index = word_ids + noise[word_ids] + 1e-6 * np.arange(len(inputs)) # 1e-6 stable sort
        key = np.argsort(index)
        outputs = [inputs[idx] for idx in key]
        return outputs

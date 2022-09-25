#coding=utf8
import os, json, math, gensim
import numpy as np
from nltk.corpus import stopwords


class NoisyChannel():

    def __init__(self, dataset, noise_type='none'):
        super(NoisyChannel, self).__init__()
        self.dataset = dataset
        self.drop, self.addition, self.shuffling = 'drop' in noise_type, 'addition' in noise_type, 'shuffling' in noise_type
        if self.drop:
            dataset_path = os.path.join('data', 'overnight') if self.dataset != 'geo' else os.path.join('data', self.dataset)
            self.nl_counter = json.load(open(os.path.join(dataset_path, self.dataset + '_count.nl'), 'r'))
            self.cf_counter = json.load(open(os.path.join(dataset_path, self.dataset + '_count.cf'), 'r'))
        if self.addition:
            model_path = os.path.join('pretrained_models', 'GoogleNews-vectors-negative300.bin.gz')
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
            self.stop_words = stopwords.words('english')


    def noisy_channel(self, inputs, input_side='cf', train_dataset=None):
        """
            inputs: raw word list
            Typically we apply three operations in sequential:
                1. importance-aware drop: drop word with probability according to counter
                2. mixed-source addition: add words from chosen candidates
                3. bi-gram shuffling: shuffle the whole sentence
        """
        outputs = inputs
        if self.drop:
            outputs = self.inject_drop_noise(outputs, input_side=input_side, max_drop_rate=0.2)
        if self.addition:
            assert train_dataset is not None
            picked_sentences = np.random.choice(np.arange(len(train_dataset)), 50, replace=False)
            picked_sentences = [train_dataset[idx].cf if input_side == 'nl' else train_dataset[idx].nl for idx in picked_sentences]
            candidate, _ = self.pick_candidate(inputs, picked_sentences)
            outputs = self.inject_addition_noise(outputs, candidate, n_gram=1, min_ratio=0.1, max_ratio=0.2)
        if self.shuffling:
            outputs = self.inject_shuffling_noise(outputs, degree=3, n_gram=2)
        return outputs


    def inject_drop_noise(self, inputs, input_side='nl', max_drop_rate=0.2):
        counter = self.nl_counter if input_side == 'nl' else self.cf_counter
        count = [counter[w] if w in counter else 0 for w in inputs]
        freq = np.array([each / float(sum(count)) for each in count])
        clipped_freq = np.clip(freq, 0., max_drop_rate)
        drop_prob = np.random.rand(len(clipped_freq))
        outputs = [w for idx, w in enumerate(inputs) if drop_prob[idx] > clipped_freq[idx]]
        if len(outputs) < 0.5 * len(inputs): # avoid dropping too much
            return inputs
        return outputs


    def pick_candidate(self, inputs, candidates):

        def sentence_sim_score(s1, s2):
            s1 = [w for w in s1 if w not in self.stop_words]
            s2 = [w for w in s2 if w not in self.stop_words]
            return self.model.wmdistance(s1, s2)

        scores = [sentence_sim_score(inputs, each) for each in candidates]
        return candidates[scores.index(min(scores))], scores.index(min(scores))


    def inject_addition_noise(self, inputs, candidate, n_gram=1, min_ratio=0.1, max_ratio=0.2):
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


    def inject_shuffling_noise(self, inputs, degree=None, n_gram=1):
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
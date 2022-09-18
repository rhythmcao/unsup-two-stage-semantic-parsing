#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.constants import BOS, EOS, PAD, UNK, DOMAINS
from utils.bleu import get_bleu_score

class Domain():

    def __init__(self):
        super(Domain, self).__init__()
        self.dataset = None

    @classmethod
    def from_dataset(self, dataset):
        if dataset in ['geo', 'scholar']:
            from utils.domain.domain_geo_scholar import GeoScholarDomain
            return GeoScholarDomain(dataset)
        elif dataset in DOMAINS:
            from utils.domain.domain_overnight import OvernightDomain
            return OvernightDomain(dataset)
        else:
            raise ValueError('Unknown domain name %s' % (dataset))

    def reverse(self, idx_list, vocab, end_mask=EOS, special_list=[BOS, EOS, PAD]):
        '''
        Change idx list to token list without special tokens.
        @args:
            1. idx_list: list of idx list, not tensor
            2. vocab: idx to token list
            3. end_mask: stop parsing when meets this token
            5. special_list: remove these tokens in sequence, list of symbols
        @return:
            token list
        '''
        seq = [[vocab[tok] for tok in tokens] for tokens in idx_list]

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        result = [trim(ex, end_mask) for ex in seq]

        def filter_special(tok):
            return tok not in special_list

        result = [list(filter(filter_special, ex)) for ex in result]
        return result

    def compare_nl(self, predictions, references):
        """
            predictions and references should be list of token list
        """
        n_best = int(len(predictions) / len(references))
        references = [[ref] for ref in references for _ in range(n_best)]
        bleu_list = list(map(get_bleu_score, predictions, references)) # sentence-level bleu score
        return bleu_list

    def compare_cf(self, predictions, references):
        """
            Use trival accuracy to evaluate canonical forms
        """
        n_best = int(len(predictions) / len(references))
        predictions = [' '.join(pred) for pred in predictions]
        references = [' '.join(ref) for ref in references]
        references = [ref for ref in references for _ in range(n_best)]
        return list(map(lambda pred, ref: 1.0 if pred == ref else 0., predictions, references))
        # references = [[ref] for ref in references for _ in range(n_best)]
        # bleu_list = []
        # for pred, ref in zip(predictions, references):
            # bleu_list.append(get_bleu_score(pred, ref, weights=(0, 0, 0.5, 0.5)))
        # return bleu_list

    def compare_lf(self, predictions, references, pick=True, variables=None):
        """
            predictions and references should be list of token list
            pick(bool): pick the first prediction without syntax or execution error if n_best > 1
        """
        predictions = self.normalize(predictions)
        references = self.normalize(references)
        n_best = int(len(predictions) / len(references))
        all_lf = predictions + references
        if variables[0] is not None: # for dataset geo or scholar
            pred_variables = [idx for idx in variables for _ in range(len(predictions))]
            all_variables = pred_variables + variables
        else:
            all_variables = None
        denotations = self.obtain_denotations(all_lf, variables=all_variables)
        predictions, references = denotations[:len(predictions)], denotations[len(predictions):]
        if pick:
            predictions, _ = self.pick_predictions(predictions, n_best)
        else:
            references = [each for each in references for _ in range(n_best)]
        return list(map(lambda x, y: 1.0 if x == y else 0.0, predictions, references))

    def normalize(self, lf_list):
        """
            Normalize each logical form, at least changes token list into string list
        """
        return [' '.join(lf) for lf in lf_list]

    def obtain_denotations(self, lf_list, variables=None):
        """
            Obtain denotations for each logical form
        """
        return lf_list

    def pick_predictions(self, pred_ans, n_best=1):
        if n_best == 1:
            return pred_ans, [i for i in range(len(pred_ans))]
        flags = self.is_valid(pred_ans)
        batches = int(len(pred_ans) / n_best)
        return_ans, return_idx = [], []
        for idx in range(batches):
            for j in range(n_best):
                if int(flags[idx * n_best + j]) == 1:
                    return_ans.append(pred_ans[idx * n_best + j])
                    return_idx.append(idx * n_best + j)
                    break
            else:
                return_ans.append(pred_ans[idx * n_best])
                return_idx.append(idx * n_best)
        return return_ans, return_idx

    def is_valid(self, ans_list):
        """
            Check whether ans is syntax or semantic invalid
            ans_list(str list): denotation list or logical form list
        """
        raise [1.0 for _ in range(len(ans_list))]

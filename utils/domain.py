#coding=utf8
import re, os, tempfile, subprocess
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utils.constants import BOS, EOS, PAD


def get_bleu_score(candidate_list, references_list, method=0, weights=(0.25, 0.25, 0.25, 0.25)):
    '''
        @args:
        if candidate_list is words list, e.g. ['which','flight']
            references_list is list of words list, e.g. [ ['which','flight'] , ['what','flight'] ]
            calculate bleu score of one sentence
        if candidate_list is list of words list, e.g. [ ['which','flight'] , ['when','to','flight'] ]
            references_list should be, e.g.
            [ [ ['which','flight'] , ['what','flight'] ] , [ ['when','to','flight'] , ['when','to','go'] ] ]
            calculate bleu score of multiple sentences, a whole corpus
        method(int): chencherry smoothing methods choice
    '''
    chencherry = SmoothingFunction()
    if len(candidate_list) == 0:
        raise ValueError('[Error]: there is no candidate sentence!')
    if type(candidate_list[0]) == str:
        return sentence_bleu(
                    references_list,
                    candidate_list,
                    weights,
                    eval('chencherry.method' + str(method))
                )
    else:
        return corpus_bleu(
                    references_list,
                    candidate_list,
                    weights,
                    eval('chencherry.method' + str(method))
                )


class Domain():

    def __init__(self, dataset):
        super(Domain, self).__init__()
        self.dataset = dataset


    def reverse(self, idx_list, vocab, end_mask=EOS, special_list=[BOS, EOS, PAD]):
        ''' Change idx list to token list, excluding special tokens.
        @args:
            1. idx_list: list of idx list, not tensor
            2. vocab: idx to token list
            3. end_mask: stop parsing when meets this token
            5. special_list: remove these tokens in sequence, list of symbols
        @return: token list
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
        """ predictions and references should be list of token list
        """
        n_best = int(len(predictions) / len(references))
        references = [[ref] for ref in references for _ in range(n_best)]
        bleu_list = list(map(get_bleu_score, predictions, references)) # sentence-level bleu score
        return bleu_list


    def compare_cf(self, predictions, references):
        """ Use trival accuracy to evaluate canonical forms
        """
        n_best = int(len(predictions) / len(references))
        predictions = [' '.join(pred) for pred in predictions]
        references = [' '.join(ref) for ref in references]
        references = [ref for ref in references for _ in range(n_best)]
        return list(map(lambda pred, ref: 1.0 if pred == ref else 0., predictions, references))


    def compare_lf(self, predictions, references, variables=[]):
        """ Use the retrieved denotations as the metric
        """
        predictions = self.normalize(predictions, variables)
        references = self.normalize(references, variables)
        n_best = len(predictions) // len(references)
        all_lf = predictions + references
        denotations = self.obtain_denotations(all_lf)
        pred_ans, ref_ans = denotations[:len(predictions)], denotations[len(predictions):]
        pred_ans = self.pick_predictions(pred_ans, n_best)
        return list(map(lambda x, y: 1.0 if x == y else 0.0, pred_ans, ref_ans))


    def normalize(self, lf_list, variables=[]):
        """ Normalize each logical form, replace PLACEHOLDER such as __state__ to entity
        """
        lf_list = [' '.join(lf) for lf in lf_list]
        replacements = [ ('(', ' ( '), (')', ' ) '), ('! ', '!'), ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld')]
        n_best = len(lf_list) // len(variables)

        def format(idx, l):
            for a, b in replacements: l = l.replace(a, b)
            l = re.sub(r'\s+', ' ', l).strip()
            vid = idx // n_best
            for k in variables[vid]: l = l.replace(' ' + k + ' ', ' ' + variables[vid][k] + ' ')
            return l

        return [format(idx, l) for idx, l in enumerate(lf_list)]


    def obtain_denotations(self, lf_list):
        """ Obtain denotations for each executable logical form
        """
        FNULL = open(os.devnull, "w")
        if self.dataset == "geo": subdomain = "geo880"
        else: subdomain = self.dataset
        tf = tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.examples')
        for line in lf_list:
            tf.write(line + '\n')
        tf.flush()
        msg = subprocess.check_output(['evaluator/overnight', subdomain, tf.name], stderr=FNULL).decode('utf8')
        tf.close()
        denotations = [self.null_to_empty_set(line.split('\t')[1]) for line in msg.split('\n') if line.startswith('targetValue\t')]
        return denotations


    def pick_predictions(self, pred_ans, n_best):
        """ Pick the first denotation which does not have runtime error if n_best > 1
        """
        if n_best == 1: return pred_ans

        flags = self.is_valid(pred_ans)
        num_samples = len(pred_ans) // n_best
        return_ans = []
        for idx in range(num_samples):
            for jdx in range(n_best):
                if int(flags[idx * n_best + jdx]) == 1:
                    return_ans.append(pred_ans[idx * n_best + jdx])
                    break
            else: return_ans.append(pred_ans[idx * n_best])
        return return_ans


    def is_valid(self, ans_list):
        """ Check whether ans is invalid.
        @args:
            ans_list(str list): denotation list or logical form list
        """
        return list(map(lambda ans: 0.0 if 'BADJAVA' in ans or 'ERROR' in ans or ans == 'null' else 1.0, ans_list))


    def null_to_empty_set(self, ans):
        """These execution errors indicate an empty set result"""
        if (ans == "BADJAVA: java.lang.RuntimeException: java.lang.NullPointerException" or ans == "BADJAVA: java.lang.RuntimeException: java.lang.RuntimeException: DB doesn't contain entity null"):
            return "(list)"
        else: return ans
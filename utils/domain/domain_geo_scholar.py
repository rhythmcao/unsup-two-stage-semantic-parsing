#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.domain.domain_overnight import OvernightDomain
import tempfile
import subprocess
import re

class GeoScholarDomain(OvernightDomain):

    def __init__(self, dataset):
        self.dataset = dataset

    def obtain_denotations(self, lf_list, variables=None):
        new_lf_list = []
        for var, lf in zip(variables, lf_list):
            new_lf = []
            for t in lf.split(' '):
                next_token = self.lexicalize_entity(var, t)
                new_lf.append(next_token)
            new_lf_list.append(" ".join(new_lf))
        new_lf_list = lf_list
        FNULL = open(os.devnull, "w")
        if self.dataset == "geo":
            evaluator_name = "evaluator/overnight"
            subdomain = "geo880"
        else:
            evaluator_name = "evaluator/scholar"
            subdomain = "external"
        tf = tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.examples')
        for line in new_lf_list:
            tf.write(line + '\n')
        tf.flush()
        msg = subprocess.check_output([evaluator_name, subdomain, tf.name], stderr=FNULL)
        msg = msg.decode('utf8')
        tf.close()
        denotations = [
            line.split('\t')[1] for line in msg.split('\n')
            if line.startswith('targetValue\t')
        ]
        return denotations

    def rep_to_empty_set(self, pred_den):
        """These execution errors indicate an empty set result"""
        if (pred_den == "BADJAVA: java.lang.RuntimeException: java.lang.NullPointerException" or pred_den == "BADJAVA: java.lang.RuntimeException: java.lang.RuntimeException: DB doesn't contain entity null"):
            return "(list)"
        else:
            return pred_den

    def lexicalize_entity(self, var, token):
        """Lexicalize an abstract entity"""
        curr_year = 2016
        if token == "year0" and (token in var or "misc0" in var):
            if token in var:
                return "number {} year".format(var[token])
            else:
                return "number {} year".format(curr_year - int(var["misc0"]))
        elif token == "misc0" and token in var:
            return "number {} count".format(var[token])
        elif token in var:
            return var[token]
        else:
            return token

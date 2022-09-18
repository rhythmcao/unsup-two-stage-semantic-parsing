#coding=utf8

import json, random, re
from utils.domain.domain_base import Domain
from itertools import product

def obtain_entity_dict(*args):
    return_dict = dict()
    for p in args:
        with open(p, 'r') as inf:
            for line in inf:
                line = line.strip()
                if line == '': continue
                *_, v = line.split('\t')
                v = json.loads(v)
                for k in v:
                    if k not in return_dict:
                        return_dict[k] = set()
                    return_dict[k].add(v[k])
    for k in return_dict:
        return_dict[k] = list(return_dict[k])
    return return_dict

def normalize(lf):
    replacements = [
        ('(', ' ( '), # make sure ( and ) must have blank space around
        (')', ' ) '),
        ('! ', '!'),
        ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
    ]
    for a, b in replacements:
        lf = lf.replace(a, b)
    # remove redundant blank spaces
    lf = re.sub(' +', ' ', lf)
    return lf

def check(domain, lf, v):
    ans = domain.obtain_denotations([lf], variables=[v])[0]
    valid = domain.is_valid([ans])[0]
    return True if valid == 1.0 else False

def try_v(domain, norm_lf, v, ent_dict):
    assignment = [k for k in v]
    num = [range(len(ent_dict[k])) for k in assignment]
    comb = list(product(*num))
    random.shuffle(comb)
    for a in comb:
        for k, idx in zip(assignment, list(a)):
            v[k] = ent_dict[k][idx]
        flag = check(domain, norm_lf, v)
        if flag:
            return v, True
    return v, False

def add(in_path, out_path, ent_dict):
    if 'geo' in in_path:
        domain = Domain.from_dataset('geo')
    else:
        domain = Domain.from_dataset('scholar')
    with open(in_path, 'r') as inf, open(out_path, 'w') as of:
        idx = 0
        for line in inf:
            line = line.strip()
            if line == '': continue
            idx += 1
            nl, cf, lf, v = line.split('\t')
            v = json.loads(v)
            norm_lf = normalize(lf)
            if len(v) != 0:
                v, flag = try_v(domain, norm_lf, v, ent_dict)
                if not flag:
                    print('Cannot find useful assignment for variables of sample:', idx)
            of.write(nl + '\t' + cf + '\t' + lf + '\t' + json.dumps(v) + '\n')

geo_dict = obtain_entity_dict('data/geo/geo_train.tsv.bak', 'data/geo/geo_dev.tsv.bak', 'data/geo/geo_test.tsv')
add('data/geo/geo_dev.tsv', 'geo_dev.tsv', geo_dict)

# scholar_dict = obtain_entity_dict('data/scholar/scholar_train.tsv.bak', 'data/scholar/scholar_dev.tsv.bak', 'data/scholar_test.tsv')
# add('data/scholar/scholar_dev.tsv', 'scholar_dev.tsv', scholar_dict)

#coding=utf8
from collections import defaultdict
import os, sys, json, random, re
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.domain import Domain

domain = Domain('geo')

KEYS = {
    '_state_': ('california', 'fb:en.state.california'),
    '_place_': ('death valley', 'fb:en.place.death_valley'),
    '_name_': ('sacramento', 'fb:en.name.sacramento'),
    '_city_': ('sacramento', 'fb:en.city.sacramento_ca'),
    '_river_': ('colorado river', 'fb:en.river.colorado')
}

nat_files = ['data/geo_granno/train_geo_nat.json', 'data/geo_granno/dev_geo_nat.json']
granno_files = ['data/geo_granno/train_geo_granno.json', 'data/geo_granno/dev_geo_granno.json']
out_granno_files = ['data/geo/geo_train.tsv', 'data/geo/geo_dev.tsv']
test_file = 'data/geo_granno/test_geo.json'
out_test_file = 'data/geo/geo_test.tsv'


def load_dataset(*files):
    examples = []
    for fp in files:
        with open(fp, 'r') as inf:
            for line in inf:
                line = line.strip()
                if line == '': continue
                ex = json.loads(line)
                examples.append(ex)
    return examples


def extract_candidates(file_list=nat_files):
    dataset = load_dataset(*file_list)
    dbs = defaultdict(set)
    for ex in dataset:
        if len(ex['variables']) > 0:
            dbs[ex['lf']].add(json.dumps(ex['variables']))
            for key in ex['variables']:
                dbs[key].add(ex['variables'][key])
    for k in dbs: dbs[k] = sorted(dbs[k])
    return dbs


def try_variables(need, dbs, lf):
    lf = lf.split(' ')
    for _ in range(1000):
        variables = {k: random.choice(dbs[k]) for k in need}
        ans = domain.obtain_denotations(domain.normalize([lf], [variables]))
        if int(domain.is_valid(ans)[0]) == 1:
            return variables
    print(f'Unresolved variables for logical form {lf}')
    return {}


def resolve_candidates(dbs, in_files=[], out_files=[], seed=999):
    random.seed(seed)
    for fp, out_fp in zip(in_files, out_files):
        dataset = load_dataset(fp)
        with open(out_fp, 'w') as of:
            for ex in tqdm(dataset, desc=f'Assign variables for file {fp}:', total=len(dataset)):
                cf, lf = ex['can'], ex['lf']
                if lf in dbs: variables = random.choice(dbs[lf])
                else:
                    need = [k for k in KEYS if ' ' + k + ' ' in lf]
                    if len(need) == 0: variables = json.dumps({})
                    else:
                        variables = json.dumps(try_variables(need, dbs, lf))
                # mapping entities in canonical utterance to the corresponding placeholders as in NL
                for key in ['_state_', '_place_', '_river_']:
                    if KEYS[key][0] in cf and key in ex['nl']:
                        cf = cf.replace(KEYS[key][0], key)
                if 'sacramento' in cf and '_city_' in ex['nl']:
                    cf = cf.replace('sacramento', '_city_')
                elif 'sacramento' in cf and '_name_' in ex['nl']:
                    cf = cf.replace('sacramento', '_name_')

                of.write(ex['nl'] + '\t' + cf + '\t' + lf + '\t' + variables + '\n')
            print(f'Write to output file: {out_fp}')


def write_test_dataset(in_file, out_file):
    dataset = load_dataset(test_file)
    with open(out_test_file, 'w') as of:
        for ex in dataset:
            of.write(ex['nl'] + '\t' + 'none' + '\t' + ex['lf'] + '\t' + json.dumps(ex['variables']) + '\n')
    print(f'Transform in_file {in_file} to out_file {out_file}')


dbs = extract_candidates(nat_files)
resolve_candidates(dbs, in_files=granno_files, out_files=out_granno_files, seed=999)
write_test_dataset(in_file=test_file, out_file=out_test_file)
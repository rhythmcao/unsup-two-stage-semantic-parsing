#coding=utf8
import sys, os

domains = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']

def read_data(fp):
    data = {}
    with open(fp, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line == '': continue
            nl, cf, lf = line.split('\t')
            nl = ' '.join([i.strip() for i in nl.split(' ') if i.strip() != ''])
            cf = ' '.join([i.strip() for i in cf.split(' ') if i.strip() != ''])
            if cf not in data:
                data[cf] = set()
            data[cf].add(nl)
    return data

def merge_data(d1, d2):
    for k in d2:
        if k not in d1:
            d1[k] = d2[k]
        else:
            for each in d2[k]:
                d1[k].add(each)
    return d1

def write_data(data, fp):
    count = 0
    with open(fp, 'w') as of:
        for cf in data:
            count += 1
            of.write('NL: ' + cf + '\n')
            for nl in data[cf]:
                of.write('CF: ' + nl + '\n')
            of.write('\n')
        of.write('Total canonical form: %d' % (count))

for each in domains:
    fp_train = os.path.join('tmp', each + '_train.tsv')
    fp_test = os.path.join('tmp', each + '_test.tsv')
    data = merge_data(read_data(fp_train), read_data(fp_test))
    out = os.path.join('tmp', each + '.log')
    write_data(data, out)


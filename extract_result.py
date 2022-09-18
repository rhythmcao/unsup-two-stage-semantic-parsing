#coding=utf8
import sys, os
import re
from collections import OrderedDict

def pick_subpath(files):
    for each in files:
        return each
        # if 'shared' in each and 'noisy' in each:
            # return each
    # raise ValueError('Cannot find that subpath !')

def aggregation(directory):
    files = os.listdir(directory)
    pattern = r'Best Test .*: (.*?)\)?$'
    cf, lf, nl, flag = -1, -1, -1, False
    picked_dir = pick_subpath(files)
    file_path = os.path.join(directory, picked_dir, 'log_train.txt')
    prev_cf, prev_lf, prev_nl = -1, -1, -1
    with open(file_path, 'r') as infile:
        for line in infile:
            if 'NEW BEST' in line and 'Unsupervised' not in line:
                acc = re.search(pattern, line).group(1)
                if '/' in acc:
                    prev_cf, prev_lf = acc.split('/')
                    prev_cf, prev_lf = float(prev_cf), float(prev_lf)
                else:
                    prev_nl = float(acc)
            if 'FINAL BEST RESULT:' in line:
                flag = True
                acc = re.search(pattern, line).group(1)
                if '/' in acc:
                    cf, lf = acc.split('/')
                    cf, lf = float(cf), float(lf)
                else:
                    nl = float(acc)
    if not flag:
        print('Missing Final Best Test Result in file %s, use best till now instead.' % (file_path))
        return prev_cf, prev_lf, prev_nl
    return cf, lf, nl

def main(directory, ratio, easy=False):

    sub_directory = os.listdir(directory)
    pattern = r'dataset_(.*)__labeled_(.*)'
    result_dict = OrderedDict()
    domains = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']
    for d in domains:
        result_dict[d] = (0, 0, 0)
    for d in sub_directory:
        domain = re.search(pattern, d).group(1)
        r = re.search(pattern, d).group(2)
        if domain not in domains:
            continue
        path = os.path.join(directory, d)
        if str(ratio) != str(r):
            continue
        cf, lf, nl = aggregation(path)
        result_dict[domain] = (cf, lf, nl)
    for d in domains:
        print('Domain %s , cf acc %.4f , lf acc %.4f , nl bleu %.4f' % (d, result_dict[d][0], result_dict[d][1], result_dict[d][2]))

    print('\n^ dataset ^ bas ^ blo ^ cal ^ hou ^ pub ^ rec ^ res ^ soc  ^ avg ^')

    string = '| nl bleu | '
    avg = 0
    for d in result_dict:
        string += '%.4f | ' % (result_dict[d][2])
        avg += result_dict[d][2]
    string += '%.4f |' % (avg / 8)
    print(string)

    string = '| cf acc | '
    avg = 0
    for d in result_dict:
        string += '%.4f | ' % (result_dict[d][0])
        avg += result_dict[d][0]
    string += '%.4f |' % (avg / 8)
    print(string)

    string = '| lf acc | '
    avg = 0
    for d in result_dict:
        string += '%.4f | ' % (result_dict[d][1])
        avg += result_dict[d][1]
    string += '%.4f |' % (avg / 8)
    print(string)

if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])

#coding=utf8
import os, sys, re

baseline_file = sys.argv[1]
our_file = sys.argv[2]
out_file = 'cases.txt'

class Instance():

    def __init__(self, inputs, ref_cf, ref_lf, pred_cf, pred_lf, cf_true, lf_true):
        super(Instance, self).__init__()
        self.inputs = inputs
        self.ref_cf = ref_cf
        self.ref_lf = ref_lf
        self.pred_cf = pred_cf
        self.pred_lf = pred_lf
        self.cf_true = True if cf_true == 'True' else False
        self.lf_true = True if lf_true == 'True' else False

def filter_instance(baseline_instances, our_instances):
    assert len(baseline_instances) == len(our_instances)
    with open(out_file, 'w') as of:
        for old, new in zip(baseline_instances, our_instances):
            if not old.lf_true and new.lf_true:
                of.write(old.inputs + '\n')
                of.write(old.ref_cf + '\n')
                of.write('Baseline pred cf: ' + old.pred_cf + '\n')
                of.write('Our pred cf: ' + new.pred_cf + '\n\n')

def load_instances(file_name):
    instance_list, it = [], 0
    # pattern = r'Epoch : (.*?)Best'
    # with open(file_name, 'r') as infile:
        # for line in infile:
            # line = line.strip()
            # Epoch : 25
            # if 'NEW BEST' not in line or 'Epoch' not in line:
                # continue
            # it = re.search(pattern, line).group(1).strip()
    # print('Best iter: %s' % (it))
    # file_name = os.path.join(os.path.dirname(file_name), 'test.iter' + it)
    with open(file_name, 'r') as infile:
        inputs, ref_cf, ref_lf, pred_cf, pred_lf = '', '', '', '', ''
        cf_true, lf_true = False, False
        index = 0
        for line in infile:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('========='):
                break
            if line.startswith('Input:'):
                inputs = line[7:]
            elif line.startswith('Ref CF:'):
                ref_cf = line[8:]
            elif line.startswith('Ref LF:'):
                ref_lf = line[8:]
            elif line.startswith('Pred CF:'):
                pred_cf = line[9:]
            elif line.startswith('Pred LF0:'):
                pred_lf = line[10:]
            elif line.startswith('CF/LF Correct:'):
                line = line[15:]
                cf_true, lf_true = line.split('/')
                instance_list.append(Instance(inputs, ref_cf, ref_lf, pred_cf, pred_lf, cf_true, lf_true))
            else:
                print('Warning: error !')
    return instance_list

baseline_instances = load_instances(baseline_file)
our_instances = load_instances(our_file)
filter_instance(baseline_instances, our_instances)

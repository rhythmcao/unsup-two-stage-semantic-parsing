#coding=utf8
import time, os, gc
from utils.example import Example
from utils.batch import get_minibatch, get_minibatch_sp
import numpy as np
import torch

class PSPSolver():

    def __init__(self, model, sp_model, vocab, sp_vocab, loss_function, optimizer, exp_path, logger, device=None, **kargs):
        super(PSPSolver, self).__init__()
        self.model, self.sp_model = model, sp_model
        self.vocab, self.sp_vocab = vocab, sp_vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
        self.best_result = { "losses": [], "iter": 0, "dev_cf_acc": 0., "dev_lf_acc": 0., "test_cf_acc": 0., "test_lf_acc": 0. }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, total_cf_acc, total_lf_acc = len(data_index), [], []
        domain = Example.domain
        self.model.eval()
        self.sp_model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, _, (raw_nl_inputs, raw_cf_outputs, raw_lf_outputs, variables) = get_minibatch(
                    data_inputs, self.vocab, task='nl2cf', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device)
                ############################ Forward NL2CF Model ############################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab.cf2id, beam_size=beam, n_best=n_best)
                    cf_predictions = results["predictions"]
                    cf_predictions = [each[0] for each in cf_predictions]
                    cf_predictions = domain.reverse(cf_predictions, self.vocab.id2cf)
                cf_acc = domain.compare_cf(cf_predictions, raw_cf_outputs)
                total_cf_acc.extend(cf_acc)
                tmp_ex_list = [Example(need=['cf', 'lf'], cf=' '.join(each), lf=' '.join(raw_lf_outputs[idx]), variables=variables[idx]) for idx, each in enumerate(cf_predictions)]
                ############################ Forward Semantic Parsing Model ############################
                inputs, lens, outputs, _, _ = get_minibatch_sp(tmp_ex_list, self.sp_vocab, device=self.device, cf_vocab=True, cf_input=True)
                with torch.no_grad():
                    results = self.sp_model.decode_batch(inputs, lens, self.sp_vocab.lf2id, beam_size=beam, n_best=n_best)
                    lf_predictions = results["predictions"]
                    lf_predictions = [pred for each in lf_predictions for pred in each]
                    lf_predictions = domain.reverse(lf_predictions, self.sp_vocab.id2lf)
                lf_acc = domain.compare_lf(lf_predictions, raw_lf_outputs, pick=True, variables=variables)
                total_lf_acc.extend(lf_acc)
                ############################ Write result to file ############################
                for idx in range(len(raw_nl_inputs)):
                    of.write("NL: " + ' '.join(raw_nl_inputs[idx]) + '\n')
                    of.write("Ref CF: " + ' '.join(raw_cf_outputs[idx]) + '\n')
                    of.write("Ref LF: " + ' '.join(raw_lf_outputs[idx]) + '\n')
                    of.write("Pred CF: " + ' '.join(cf_predictions[idx]) + '\n')
                    for i in range(n_best):
                        of.write("Pred LF" + str(i) + ": " + ' '.join(lf_predictions[n_best * idx + i]) + '\n')
                    of.write("CF/LF Correct: " + ("True" if cf_acc[idx] == 1 else "False") + '/' + ("True" if lf_acc[idx] == 1 else "False") + '\n\n')
            cf_acc = sum(total_cf_acc) / float(len(total_cf_acc))
            lf_acc = sum(total_lf_acc) / float(len(total_lf_acc))
            of.write('Overall accuracy for CF and LF is %.4f/%.4f.' % (cf_acc, lf_acc))
        return cf_acc, lf_acc

    def train_and_decode(self, train_dataset, dev_dataset, test_dataset, batchSize=16, test_batchSize=128,
            max_epoch=100, beam=5, n_best=1):
        train_data_index = np.arange(len(train_dataset))
        nsentences = len(train_data_index)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_data_index)
            losses = []
            self.model.train()
            for j in range(0, nsentences, batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, out_lens, _ = get_minibatch(
                    train_dataset, self.vocab, task='nl2cf', data_index=train_data_index,
                    index=j, batch_size=batchSize, device=self.device)
                ############################ Forward Model ############################
                self.model.zero_grad()
                batch_scores = self.model(inputs, lens, outputs[:, :-1])
                batch_loss = self.loss_function(batch_scores, outputs[:, 1:])
                losses.append(batch_loss.item())
                batch_loss.backward()
                self.model.pad_embedding_grad_zero()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            epoch_loss = np.sum(losses, axis=0)
            self.best_result['losses'].append(epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' \
                                % (i, time.time() - start_time, epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_cf_acc, dev_lf_acc = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Dev Evaluation:\tEpoch : %d\tTime : %.4fs\tCF Acc : %.4f\tLF Acc : %.4f' \
                                % (i, time.time() - start_time, dev_cf_acc, dev_lf_acc))
            start_time = time.time()
            test_cf_acc, test_lf_acc = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Test Evaluation:\tEpoch : %d\tTime : %.4fs\tCF Acc : %.4f\tLF Acc : %.4f' \
                                % (i, time.time() - start_time, test_cf_acc, test_lf_acc))

            ######################## Pick best result on dev and save #####################
            if dev_cf_acc > self.best_result['dev_cf_acc']:# and dev_lf_acc > self.best_result['dev_lf_acc']:
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.best_result['iter'] = i
                self.best_result['dev_cf_acc'], self.best_result['test_cf_acc'] = dev_cf_acc, test_cf_acc
                self.best_result['dev_lf_acc'], self.best_result['test_lf_acc'] = dev_lf_acc, test_lf_acc
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid Acc (CF/LF) : %.4f/%.4f;\tBest Test Acc (CF/LF) : %.4f/%.4f' % (i, dev_cf_acc, dev_lf_acc, test_cf_acc, test_lf_acc))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (CF/LF Acc : %.4f/%.4f)\tBest Test (CF/LF Acc : %.4f/%.4f)'
                % (self.best_result['iter'], self.best_result['dev_cf_acc'], self.best_result['dev_lf_acc'], self.best_result['test_cf_acc'], self.best_result['test_lf_acc']))
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))

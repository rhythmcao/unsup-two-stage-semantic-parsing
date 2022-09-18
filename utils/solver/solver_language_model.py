# coding=utf8
import os, sys, time, gc
import numpy as np
import torch
from utils.batch import get_minibatch

class LMSolver():
    '''
        For traditional RNN-based Language Model
    '''
    def __init__(self, nl_model, cf_model, lm_vocab, loss_function, optimizer, exp_path, logger, device='cpu'):
        super(LMSolver, self).__init__()
        self.nl_model, self.cf_model = nl_model, cf_model
        self.vocab = lm_vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
        self.best_result = {
            "nl_losses": [], "cf_losses": [],
            "nl_iter": 0, "cf_iter": 0,
            "dev_nl_ppl": float('inf'), "test_nl_ppl": float('inf'),
            "dev_cf_ppl": float("inf"), "test_cf_ppl": float("inf")
        }

    def decode(self, data_inputs, output_path, test_batchSize):
        data_index = np.arange(len(data_inputs))
        ########################### Evaluation Phase ############################
        self.nl_model.eval()
        self.cf_model.eval()
        with open(output_path, 'w') as f:
            eval_loss, length_list = [], []
            for j in range(0, len(data_index), test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, raw_inputs = get_minibatch(data_inputs, self.vocab, task='language_model',
                    data_index=data_index, index=j, batch_size=test_batchSize, device=self.device, side='nl')
                length_list.extend((lens - 1).tolist())
                ########################## Calculate Sentence PPL #######################
                with torch.no_grad():
                    scores = self.nl_model(inputs, lens) # bsize, seq_len, voc_size
                    batch_loss = self.loss_function['nl'](scores, inputs[:, 1:]).item()
                    eval_loss.append(batch_loss)
                    norm_log_prob = self.nl_model.sent_logprobability(inputs, lens).cpu().tolist()

                ############################# Writing Result to File ###########################
                for idx in range(len(inputs)):
                    f.write('Natural Language: ' + ' '.join(raw_inputs[idx]) + '\n')
                    f.write('NormLogProb: ' + str(norm_log_prob[idx]) + '\n')
                    current_ppl = np.exp(- norm_log_prob[idx])
                    f.write('PPL: ' + str(current_ppl) + '\n\n')

            ########################### Calculate Corpus PPL ###########################
            word_count = np.sum(length_list, axis=0)
            eval_loss = np.sum(eval_loss, axis=0)
            final_nl_ppl = np.exp(eval_loss / word_count)

            eval_loss, length_list = [], []
            for j in range(0, len(data_index), test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, raw_inputs = get_minibatch(data_inputs, self.vocab, task='language_model',
                    data_index=data_index, index=j, batch_size=test_batchSize, device=self.device, side='cf')
                length_list.extend((lens - 1).tolist())
                ########################## Calculate Sentence PPL #######################
                with torch.no_grad():
                    scores = self.cf_model(inputs, lens) # bsize, seq_len, voc_size
                    batch_loss = self.loss_function['cf'](scores, inputs[:, 1:]).item()
                    eval_loss.append(batch_loss)
                    norm_log_prob = self.cf_model.sent_logprobability(inputs, lens).cpu().tolist()

                ############################# Writing Result to File ###########################
                for idx in range(len(inputs)):
                    f.write('Canonical Form: ' + ' '.join(raw_inputs[idx]) + '\n')
                    f.write('NormLogProb: ' + str(norm_log_prob[idx]) + '\n')
                    current_ppl = np.exp(- norm_log_prob[idx])
                    f.write('PPL: ' + str(current_ppl) + '\n\n')

            ########################### Calculate Corpus PPL ###########################
            word_count = np.sum(length_list, axis=0)
            eval_loss = np.sum(eval_loss, axis=0)
            final_cf_ppl = np.exp(eval_loss / word_count)
            f.write('Overall NL/CF ppl: %.4f/%.4f.' % (final_nl_ppl, final_cf_ppl))
        return final_nl_ppl, final_cf_ppl

    def train_and_decode(self, train_inputs, dev_inputs, test_inputs, batchSize=16, test_batchSize=128, max_epoch=100):
        train_data_index = np.arange(len(train_inputs))
        nsentences = len(train_data_index)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_data_index)
            nl_losses, cf_losses = [], []
            self.nl_model.train()
            self.cf_model.train()
            for j in range(0, nsentences, batchSize):
                self.optimizer.zero_grad()
                ###################### Obtain minibatch data ######################
                inputs, lens, _ = get_minibatch(train_inputs, self.vocab, task='language_model',
                    data_index=train_data_index, index=j, batch_size=batchSize, device=self.device, side='nl')
                ############################ Forward Model ############################
                batch_scores = self.nl_model(inputs, lens)
                batch_loss = self.loss_function["nl"](batch_scores, inputs[:, 1:])
                nl_losses.append(batch_loss.item())
                batch_loss.backward()
                self.nl_model.pad_embedding_grad_zero()

                ###################### Obtain minibatch data ######################
                inputs, lens, _ = get_minibatch(train_inputs, self.vocab, task='language_model',
                    data_index=train_data_index, index=j, batch_size=batchSize, device=self.device, side='cf')
                ############################ Forward Model ############################
                batch_scores = self.cf_model(inputs, lens)
                batch_loss = self.loss_function["cf"](batch_scores, inputs[:, 1:])
                cf_losses.append(batch_loss.item())
                batch_loss.backward()
                self.cf_model.pad_embedding_grad_zero()

                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            nl_epoch_loss, cf_epoch_loss = np.sum(nl_losses, axis=0), np.sum(cf_losses, axis=0)
            self.best_result['nl_losses'].append(nl_epoch_loss)
            self.best_result['cf_losses'].append(cf_epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss of NL/CF Language Model : %.5f/%.5f' \
                                % (i, time.time() - start_time, nl_epoch_loss, cf_epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            # whether evaluate later after training for some epochs
            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_nl_ppl, dev_cf_ppl = self.decode(dev_inputs, os.path.join(self.exp_path, 'valid.iter' + str(i)), test_batchSize)
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tNL/CF ppl : %.4f/%.4f' % (i, time.time() - start_time, dev_nl_ppl, dev_cf_ppl))
            start_time = time.time()
            test_nl_ppl, test_cf_ppl = self.decode(test_inputs, os.path.join(self.exp_path, 'test.iter' + str(i)), test_batchSize)
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tNL/CF ppl : %.4f/%.4f' % (i, time.time() - start_time, test_nl_ppl, test_cf_ppl))

            ######################## Pick best result and save #####################
            if dev_nl_ppl < self.best_result['dev_nl_ppl']:
                self.nl_model.save_model(os.path.join(self.exp_path, 'nl_model.pkl'))
                self.best_result['nl_iter'] = i
                self.best_result['dev_nl_ppl'], self.best_result['test_nl_ppl'] = dev_nl_ppl, test_nl_ppl
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid NL ppl : %.4f;\tBest Test NL ppl : %.4f' % (i, dev_nl_ppl, test_nl_ppl))
            if dev_cf_ppl < self.best_result['dev_cf_ppl']:
                self.cf_model.save_model(os.path.join(self.exp_path, 'cf_model.pkl'))
                self.best_result['cf_iter'] = i
                self.best_result['dev_cf_ppl'], self.best_result['test_cf_ppl'] = dev_cf_ppl, test_cf_ppl
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid CF ppl : %.4f;\tBest Test CF ppl : %.4f' % (i, dev_cf_ppl, test_cf_ppl))

        ######################## Reload best model for later usage #####################
        self.logger.info('NL FINAL BEST RESULT: \tEpoch : %d\tBest Valid (ppl : %.4f)\tBest Test (ppl : %.4f) '
            % (self.best_result['nl_iter'], self.best_result['dev_nl_ppl'], self.best_result['test_nl_ppl']))
        self.logger.info('CF FINAL BEST RESULT: \tEpoch : %d\tBest Valid (ppl : %.4f)\tBest Test (ppl : %.4f) '
            % (self.best_result['cf_iter'], self.best_result['dev_cf_ppl'], self.best_result['test_cf_ppl']))
        self.nl_model.load_model(os.path.join(self.exp_path, 'nl_model.pkl'))
        self.cf_model.load_model(os.path.join(self.exp_path, 'cf_model.pkl'))


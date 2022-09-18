#coding=utf8
import time, os, gc
from utils.example import Example
from utils.batch import get_minibatch
import numpy as np
import torch

class MTSPSolver():

    def __init__(self, sp_model, nl_model, vocab, loss_function, optimizer, exp_path, logger, device=None, **kargs):
        super(MTSPSolver, self).__init__()
        self.sp_model, self.nl_model = sp_model, nl_model
        self.vocab = vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
        self.best_result = { "losses": [], "iter": 0, "dev_cf_acc": 0., "dev_nl_acc": 0., "test_cf_acc": 0., "test_nl_acc": 0. }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, cf_total, nl_total = len(data_index), [], []
        domain = Example.domain
        self.sp_model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, _, (raw_cf_inputs, raw_outputs, variables) = get_minibatch(
                    data_inputs, self.vocab, task='semantic_parsing', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device, cf_vocab=False, cf_input=True)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.sp_model.decode_batch(inputs, lens, self.vocab.lf2id, beam_size=beam, n_best=n_best)
                    cf_predictions = results["predictions"]
                    cf_predictions = [pred for each in cf_predictions for pred in each]
                    cf_predictions = domain.reverse(cf_predictions, self.vocab.id2lf)
                cf_acc = domain.compare_lf(cf_predictions, raw_outputs, pick=True, variables=variables)
                cf_total.extend(cf_acc)
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, _, (raw_nl_inputs, _, _) = get_minibatch(
                    data_inputs, self.vocab, task='semantic_parsing', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device, cf_vocab=False, cf_input=False)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.sp_model.decode_batch(inputs, lens, self.vocab.lf2id, beam_size=beam, n_best=n_best)
                    nl_predictions = results["predictions"]
                    nl_predictions = [pred for each in nl_predictions for pred in each]
                    nl_predictions = domain.reverse(nl_predictions, self.vocab.id2lf)
                nl_acc = domain.compare_lf(nl_predictions, raw_outputs, pick=True, variables=variables)
                nl_total.extend(nl_acc)
                ############################ Write result to file ############################
                for idx in range(len(raw_outputs)):
                    of.write("Ref LF: " + ' '.join(raw_outputs[idx]) + '\n')
                    of.write("CF: " + ' '.join(raw_cf_inputs[idx]) + '\n')
                    for i in range(n_best):
                        of.write("Pred LF" + str(i) + ": " + ' '.join(cf_predictions[n_best * idx + i]) + '\n')
                    of.write("Correct: " + ("True" if cf_acc[idx] == 1 else "False") + '\n')
                    of.write("NL: " + ' '.join(raw_nl_inputs[idx]) + '\n')
                    for i in range(n_best):
                        of.write("Pred LF" + str(i) + ": " + ' '.join(nl_predictions[n_best * idx + i]) + '\n')
                    of.write("Correct: " + ("True" if nl_acc[idx] == 1 else "False") + '\n\n')
            cf_acc = sum(cf_total) / float(len(cf_total))
            nl_acc = sum(nl_total) / float(len(nl_total))
            of.write('Overall accuracy for CF/NL input is %.4f/%.4f.' % (cf_acc, nl_acc))
        return cf_acc, nl_acc

    def train_and_decode(self, train_dataset, dev_dataset, test_dataset, batchSize=16, test_batchSize=128,
            max_epoch=100, beam=5, n_best=1):
        sp_train_data_index = np.arange(len(train_dataset))
        nl_train_data_index = np.arange(len(train_dataset))
        nsentences = len(train_dataset)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(sp_train_data_index)
            np.random.shuffle(nl_train_data_index)
            losses = {}
            losses['sp'], losses['nl'] = [], []
            self.sp_model.train()
            self.nl_model.train()
            for j in range(0, nsentences, batchSize):
                self.sp_model.zero_grad()
                self.nl_model.zero_grad()
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, out_lens, _ = get_minibatch(
                    train_dataset, self.vocab, task='semantic_parsing', data_index=sp_train_data_index,
                    index=j, batch_size=batchSize, device=self.device, cf_vocab=False, cf_input=True)
                ############################ Forward Model ############################
                batch_scores = self.sp_model(inputs, lens, outputs[:, :-1])
                batch_loss = self.loss_function['sp'](batch_scores, outputs[:, 1:])
                losses['sp'].append(batch_loss.item())
                batch_loss.backward()

                inputs, lens, outputs, out_lens, _ = get_minibatch(
                    train_dataset, self.vocab, task='multi_task', data_index=nl_train_data_index,
                    index=j, batch_size=batchSize, device=self.device, train_dataset=train_dataset)
                ############################ Forward Model ############################
                batch_scores = self.nl_model(inputs, lens, outputs[:, :-1])
                batch_loss = self.loss_function['nl'](batch_scores, outputs[:, 1:])
                losses['nl'].append(batch_loss.item())
                batch_loss.backward()

                self.sp_model.pad_embedding_grad_zero()
                self.nl_model.pad_embedding_grad_zero()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            sp_epoch_loss, nl_epoch_loss = np.sum(losses['sp']), np.sum(losses['nl'])
            self.best_result['losses'].append((sp_epoch_loss, nl_epoch_loss))
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t SP/NL Loss: %.5f/%.5f' \
                                % (i, time.time() - start_time, sp_epoch_loss, nl_epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_cf_acc, dev_nl_acc = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Dev Evaluation:\tEpoch : %d\tTime : %.4fs\tCF/NL Acc : %.4f/%.4f' \
                                % (i, time.time() - start_time, dev_cf_acc, dev_nl_acc))
            start_time = time.time()
            test_cf_acc, test_nl_acc = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Test Evaluation:\tEpoch : %d\tTime : %.4fs\tCF/NL Acc : %.4f/%.4f' \
                                % (i, time.time() - start_time, test_cf_acc, test_nl_acc))

            ######################## Pick best result on dev and save #####################
            if dev_cf_acc > self.best_result['dev_cf_acc']:
                self.sp_model.save_model(os.path.join(self.exp_path, 'sp_model.pkl'))
                self.nl_model.save_model(os.path.join(self.exp_path, 'nl_model.pkl'))
                self.best_result['iter'] = i
                self.best_result['dev_cf_acc'], self.best_result['test_cf_acc'] = dev_cf_acc, test_cf_acc
                self.best_result['dev_nl_acc'], self.best_result['test_nl_acc'] = dev_nl_acc, test_nl_acc
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid CF/NL Acc : %.4f/%.4f;\tBest Test CF/NL Acc : %.4f/%.4f' \
                    % (i, dev_cf_acc, dev_nl_acc, test_cf_acc, test_nl_acc))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (CF/NL Acc : %.4f/%.4f)\tBest Test (CF/NL Acc : %.4f/%.4f)'
                % (self.best_result['iter'], self.best_result['dev_cf_acc'], self.best_result['dev_nl_acc'], self.best_result['test_cf_acc'], self.best_result['test_nl_acc']))
        self.sp_model.load_model(os.path.join(self.exp_path, 'sp_model.pkl'))
        self.nl_model.load_model(os.path.join(self.exp_path, 'nl_model.pkl'))

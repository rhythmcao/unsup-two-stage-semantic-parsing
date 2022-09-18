# coding=utf8
import os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn
from utils.batch import get_minibatch

class CLSFSolver():
    def __init__(self, model, vocab, loss_function, optimizer, exp_path, logger, device=None, **kargs):
        super(CLSFSolver, self).__init__()
        self.model = model
        self.vocab = vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
        self.best_result = {"losses": [], "iter": 0, "dev_acc": 0., "test_acc": 0.}

    def decode(self, data_inputs, output_path, test_batchSize):
        data_index = np.arange(len(data_inputs))
        nsentences = len(data_index)
        self.model.eval()
        total_acc = []
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, outputs, raw_inputs = get_minibatch(
                    data_inputs, self.vocab, task='discriminator', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.model(inputs)
                    predictions = (results >= 0.5).long().tolist()
                acc = list(map(lambda x, y: 1 if x == y else 0, predictions, outputs))
                total_acc.extend(acc)
                ############################ Write result to file ############################
                for idx in range(len(raw_inputs)):
                    of.write('Input: ' + ' '.join(raw_inputs[idx]) + '\n')
                    of.write('Ref: ' + ('CF' if outputs[idx] else 'NL') + '\n')
                    of.write('Pred: ' + ('CF' if predictions[idx] else 'NL') + '\n')
                    of.write('Correct: ' + ('True' if acc[idx] else 'False') + '\n\n')
            acc = sum(total_acc) / float(len(total_acc))
            of.write('Overall accuracy for style classification is %.4f.' % (acc))
        return acc

    def train_and_decode(self, train_dataset, dev_dataset, test_dataset, batchSize=50, test_batchSize=128, max_epoch=100):
        train_data_index = np.arange(len(train_dataset))
        nsentences = len(train_data_index)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_data_index)
            losses = []
            self.model.train()
            for j in range(0, nsentences, batchSize):
                self.model.zero_grad()
                ###################### Obtain minibatch data ######################
                inputs, outputs, _ = get_minibatch(
                    train_dataset, self.vocab, task='discriminator', data_index=train_data_index,
                    index=j, batch_size=batchSize, device=self.device)
                ############################ Forward Model ############################
                pred = self.model(inputs)
                batch_loss = self.loss_function(pred, outputs)
                losses.append(batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            epoch_loss = np.sum(losses, axis=0)
            self.best_result['losses'].append(epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' % (i, time.time() - start_time, epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_acc = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)), test_batchSize)
            self.logger.info('Dev Evaluation:\tEpoch : %d\tTime : %.4fs\tAcc : %.4f' \
                             % (i, time.time() - start_time, dev_acc))
            start_time = time.time()
            test_acc = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)), test_batchSize)
            self.logger.info('Test Evaluation:\tEpoch : %d\tTime : %.4fs\t Acc : %.4f' \
                             % (i, time.time() - start_time, test_acc))

            ######################## Pick best result on dev and save #####################
            if dev_acc > self.best_result['dev_acc']:
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.best_result['iter'] = i
                self.best_result['dev_acc'], self.best_result['test_acc'] = dev_acc, test_acc
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid Acc : %.4f;\tBest Test Acc : %.4f' \
                    % (i, dev_acc, test_acc))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (Acc : %.4f)\tBest Test (Acc : %.4f)'
            % (self.best_result['iter'], self.best_result['dev_acc'], self.best_result['test_acc']))
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))

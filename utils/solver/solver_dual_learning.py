#coding=utf8
import os, sys, time, gc, json
import numpy as np
import torch
from utils.constants import *
from utils.batch import get_minibatch, get_minibatch_sp, get_minibatch_unlabeled_nl2cf, get_minibatch_unlabeled_cf2nl
from utils.example import Example

class DualLearningSolver():
    '''
        For Dual Learning Solver
    '''
    def __init__(self, model, sp_model, vocab, loss_function, optimizer, exp_path, logger, device):
        super(DualLearningSolver, self).__init__()
        self.model, self.sp_model = model, sp_model
        self.vocab = vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
        self.best_result = {
            "nl2cf_losses": [], "cf2nl_losses": [], "iter": 0,
            "dev_cf_acc": 0., "dev_lf_acc": 0., "dev_nl_bleu": 0.,
            "test_cf_acc": 0., "test_lf_acc": 0., "test_nl_bleu": 0.,
            "self_dev_cf_acc": 0., "self_dev_lf_acc": 0., "self_dev_nl_bleu": 0.
        }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, total_cf_acc, total_lf_acc, total_nl_bleu = len(data_index), [], [], []
        domain = Example.domain
        self.model.eval()
        self.sp_model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, _, (raw_cf_inputs, raw_cf_outputs, raw_lf_outputs, variables) = get_minibatch(
                    data_inputs, self.vocab["nl2cf"], task='nl2cf', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device["nl2cf"])
                ############################ Forward NL2CF Model ############################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab["nl2cf"].cf2id, task="nl2cf", beam_size=beam, n_best=n_best)
                    cf_predictions = results["predictions"]
                    cf_predictions = [each[0] for each in cf_predictions]
                    cf_predictions = domain.reverse(cf_predictions, self.vocab["nl2cf"].id2cf)
                cf_acc = domain.compare_cf(cf_predictions, raw_cf_outputs)
                total_cf_acc.extend(cf_acc)
                tmp_ex_list = [Example(need=['cf', 'lf'], cf=' '.join(each), lf=' '.join(raw_lf_outputs[idx]), variables=variables[idx]) for idx, each in enumerate(cf_predictions)]
                ############################ Forward Semantic Parsing Model ############################
                inputs, lens, outputs, _, _ = get_minibatch_sp(tmp_ex_list, self.vocab["sp"], device=self.device['sp'], cf_vocab=True, cf_input=True)
                with torch.no_grad():
                    results = self.sp_model.decode_batch(inputs, lens, self.vocab["sp"].lf2id, beam_size=beam, n_best=n_best)
                    lf_predictions = results["predictions"]
                    lf_predictions = [pred for each in lf_predictions for pred in each]
                    lf_predictions = domain.reverse(lf_predictions, self.vocab["sp"].id2lf)
                lf_acc = domain.compare_lf(lf_predictions, raw_lf_outputs, pick=True, variables=variables)
                total_lf_acc.extend(lf_acc)
                ############################ Write result to file ############################
                for idx in range(len(raw_cf_inputs)):
                    of.write("Input: " + ' '.join(raw_cf_inputs[idx]) + '\n')
                    of.write("Ref CF: " + ' '.join(raw_cf_outputs[idx]) + '\n')
                    of.write("Ref LF: " + ' '.join(raw_lf_outputs[idx]) + '\n')
                    of.write("Pred CF: " + ' '.join(cf_predictions[idx]) + '\n')
                    for i in range(n_best):
                        of.write("Pred LF" + str(i) + ": " + ' '.join(lf_predictions[n_best * idx + i]) + '\n')
                    of.write("CF/LF Correct: " + ("True" if cf_acc[idx] == 1 else "False") + '/' + ("True" if lf_acc[idx] == 1 else "False") + '\n\n')

            of.write('=' * 50 + '\n\n')

            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, _, (raw_nl_inputs, raw_nl_outputs) = get_minibatch(
                    data_inputs, self.vocab["cf2nl"], task='cf2nl', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device["cf2nl"])
                ############################ Forward CF2NL Model ############################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab["cf2nl"].nl2id, task="cf2nl", beam_size=beam, n_best=n_best)
                    nl_predictions = results["predictions"]
                    nl_predictions = [each[0] for each in nl_predictions]
                    nl_predictions = domain.reverse(nl_predictions, self.vocab["cf2nl"].id2nl)
                nl_bleu = domain.compare_nl(nl_predictions, raw_nl_outputs)
                total_nl_bleu.extend(nl_bleu)
                ############################ Write result to file ############################
                for idx in range(len(raw_nl_inputs)):
                    of.write("Input: " + ' '.join(raw_nl_inputs[idx]) + '\n')
                    of.write("Ref NL: " + ' '.join(raw_nl_outputs[idx]) + '\n')
                    of.write("Pred NL: " + ' '.join(nl_predictions[idx]) + '\n')
                    of.write("Bleu: " + str(nl_bleu[idx]) + '\n\n')

            cf_acc = sum(total_cf_acc) / float(len(total_cf_acc))
            lf_acc = sum(total_lf_acc) / float(len(total_lf_acc))
            nl_bleu = sum(total_nl_bleu) / float(len(total_nl_bleu))
            of.write('Overall CF/LF accuracy for NL2CF is %.4f/%.4f.\n' % (cf_acc, lf_acc))
            of.write('Overall bleu score for CF2NL is %.4f.' % (nl_bleu))
        return cf_acc, lf_acc, nl_bleu

    def unsupervised_evaluation(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, total_cf_acc, total_lf_acc, total_nl_bleu = len(data_index), [], [], []
        domain = Example.domain
        self.model.eval()
        self.sp_model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, raw_inputs = get_minibatch(
                    data_inputs, self.vocab["nl2cf"], task='unlabeled_nl2cf', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device["nl2cf"])
                ############################ Forward NL2CF Model ############################
                with torch.no_grad():
                    results = self.model.nl2cf_model.decode_batch(inputs, lens, self.vocab["nl2cf"].cf2id, beam_size=beam, n_best=n_best)
                    cf_predictions = results["predictions"]
                    cf_predictions = [each[0] for each in cf_predictions]
                    cf_predictions = domain.reverse(cf_predictions, self.vocab["nl2cf"].id2cf)
                tmp_ex_list = [Example(need=['cf'], cf=' '.join(each)) for each in cf_predictions]
                ############################ Forward CF2NL Model ############################
                inputs, lens, _ = get_minibatch_unlabeled_cf2nl(tmp_ex_list, self.vocab["cf2nl"], device=self.device["cf2nl"])
                with torch.no_grad():
                    results = self.model.cf2nl_model.decode_batch(inputs, lens, self.vocab["cf2nl"].nl2id, beam_size=beam, n_best=n_best)
                    nl_predictions = results["predictions"]
                    nl_predictions = [each[0] for each in nl_predictions]
                    nl_predictions = domain.reverse(nl_predictions, self.vocab["cf2nl"].id2nl)
                nl_bleu = domain.compare_nl(nl_predictions, raw_inputs)
                total_nl_bleu.extend(nl_bleu)
                ############################ Write result to file ############################
                for idx in range(len(raw_inputs)):
                    of.write("Input NL: " + ' '.join(raw_inputs[idx]) + '\n')
                    of.write("Pred CF: " + ' '.join(cf_predictions[idx]) + '\n')
                    of.write("Pred NL: " + ' '.join(nl_predictions[idx]) + '\n')
                    of.write("Bleu: " + str(nl_bleu[idx]) + '\n\n')

            of.write('=' * 50 + '\n\n')

            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, (raw_inputs, raw_lf_inputs, variables) = get_minibatch(
                    data_inputs, self.vocab["cf2nl"], task='unlabeled_cf2nl', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device["cf2nl"])
                ############################ Forward CF2NL Model ############################
                with torch.no_grad():
                    results = self.model.cf2nl_model.decode_batch(inputs, lens, self.vocab["cf2nl"].nl2id, beam_size=beam, n_best=n_best)
                    nl_predictions = results["predictions"]
                    nl_predictions = [each[0] for each in nl_predictions]
                    nl_predictions = domain.reverse(nl_predictions, self.vocab["cf2nl"].id2nl)
                tmp_ex_list = [Example(need=['nl'], nl=' '.join(each)) for each in nl_predictions]
                ############################ Forward NL2CF Model ############################
                inputs, lens, _ = get_minibatch_unlabeled_nl2cf(tmp_ex_list, self.vocab["nl2cf"], device=self.device["nl2cf"])
                with torch.no_grad():
                    results = self.model.nl2cf_model.decode_batch(inputs, lens, self.vocab["nl2cf"].cf2id, beam_size=beam, n_best=n_best)
                    cf_predictions = results["predictions"]
                    cf_predictions = [each[0] for each in cf_predictions]
                    cf_predictions = domain.reverse(cf_predictions, self.vocab["nl2cf"].id2cf)
                cf_acc = domain.compare_cf(cf_predictions, raw_inputs)
                total_cf_acc.extend(cf_acc)
                ############################ Forward Naive Semantic Parsing Model ############################
                tmp_ex_list = [Example(need=['cf', 'lf'], cf=' '.join(each), lf=' '.join(raw_lf_inputs[idx]), variables=variables[idx]) for idx, each in enumerate(cf_predictions)]
                inputs, lens, outputs, _, _ = get_minibatch_sp(tmp_ex_list, self.vocab['sp'], device=self.device['sp'], cf_vocab=True, cf_input=True)
                with torch.no_grad():
                    results = self.sp_model.decode_batch(inputs, lens, self.vocab['sp'].lf2id, beam_size=beam, n_best=n_best)
                    lf_predictions = results["predictions"]
                    lf_predictions = [pred for each in lf_predictions for pred in each]
                    lf_predictions = domain.reverse(lf_predictions, self.vocab['sp'].id2lf)
                lf_acc = domain.compare_lf(lf_predictions, raw_lf_inputs, pick=True, variables=variables)
                total_lf_acc.extend(lf_acc)
                ############################ Write result to file ############################
                for idx in range(len(raw_inputs)):
                    of.write("Input CF: " + ' '.join(raw_inputs[idx]) + '\n')
                    of.write("Input LF: " + ' '.join(raw_lf_inputs[idx]) + '\n')
                    of.write("Pred NL: " + ' '.join(nl_predictions[idx]) + '\n')
                    of.write("Pred CF: " + ' '.join(cf_predictions[idx]) + '\n')
                    of.write("Pred LF: " + ' '.join(lf_predictions[idx]) + '\n')
                    of.write("CF/LF Acc: " + str(cf_acc[idx]) + '/' + str(lf_acc[idx]) + '\n\n')

            nl_bleu = sum(total_nl_bleu) / float(len(total_nl_bleu))
            cf_acc = sum(total_cf_acc) / float(len(total_cf_acc))
            lf_acc = sum(total_lf_acc) / float(len(total_lf_acc))
            of.write('Overall NL bleu score for nl --> cf --> nl is %.4f.\n' % (nl_bleu))
            of.write('Overall CF/LF accuracy for cf --> nl --> cf is %.4f/%.4f.' % (cf_acc, lf_acc))
        return nl_bleu, cf_acc, lf_acc

    def generate_pseudo_samples(self, batch_size, beam, nl_samples=None, cf_samples=None):
        nl_pseudo_samples, cf_pseudo_samples = [], []
        if nl_samples:
            nsentences = len(nl_samples)
            data_index = np.arange(nsentences)
            self.model.nl2cf_model.eval()
            for j in range(0, nsentences, batch_size):
                inputs, lens, raw_in = get_minibatch(nl_samples, self.vocab['nl2cf'], task='unlabeled_nl2cf', data_index=data_index, index=j, batch_size=batch_size, device=self.device['nl2cf'])
                with torch.no_grad():
                    results = self.model.nl2cf_model.decode_batch(inputs, lens, self.vocab['nl2cf'].cf2id, beam_size=beam, n_best=1)
                    cf_predictions = results['predictions']
                    cf_predictions = [hyp[0] for hyp in cf_predictions]
                    cf_predictions = Example.domain.reverse(cf_predictions, self.vocab['nl2cf'].id2cf)
                nl_pseudo_samples.extend([Example(need=['cf', 'nl'], cf=' '.join(each), nl=' '.join(raw_in[idx])) for idx, each in enumerate(cf_predictions)])
        if cf_samples:
            nsentences = len(cf_samples)
            data_index = np.arange(nsentences)
            self.model.cf2nl_model.eval()
            for j in range(0, nsentences, batch_size):
                inputs, lens, (raw_in, _, variables) = get_minibatch(cf_samples, self.vocab['cf2nl'], task='unlabeled_cf2nl', data_index=data_index, index=j, batch_size=batch_size, device=self.device['cf2nl'])
                with torch.no_grad():
                    results = self.model.cf2nl_model.decode_batch(inputs, lens, self.vocab['cf2nl'].nl2id, beam_size=beam, n_best=1)
                    nl_predictions = results['predictions']
                    nl_predictions = [hyp[0] for hyp in nl_predictions]
                    nl_predictions = Example.domain.reverse(nl_predictions, self.vocab['cf2nl'].id2nl)
                cf_pseudo_samples.extend([Example(need=['nl', 'cf'], nl=' '.join(each), cf=' '.join(raw_in[idx]), variables=variables[idx]) for idx, each in enumerate(nl_predictions)])
        return nl_pseudo_samples, cf_pseudo_samples

    def train_and_decode(self, unlabeled_nl, unlabeled_cf, labeled_data, dev_dataset, test_dataset, batchSize, test_batchSize,
            scheme='bt+drl', max_epoch=100, beam=5, n_best=1):
        if 'drl' in scheme:
            unlabeled_nl_index = np.arange(len(unlabeled_nl))
            unlabeled_cf_index = np.arange(len(unlabeled_cf))
        if 'dae' in scheme:
            self_train_nl_index = np.arange(len(unlabeled_nl))
            self_train_cf_index = np.arange(len(unlabeled_cf))
        if 'bt' in scheme:
            pseudo_nl_index = np.arange(len(unlabeled_cf))
            pseudo_cf_index = np.arange(len(unlabeled_nl))
        if labeled_data != []:
            labeled_train_index = np.arange(len(labeled_data))
        nsentences = max([len(unlabeled_nl), len(unlabeled_cf)])
        for i in range(max_epoch):
            if 'bt' in scheme:
                cf_pseudo_samples, nl_pseudo_samples = self.generate_pseudo_samples(test_batchSize, beam, unlabeled_nl, unlabeled_cf)
            ########################### Training Phase ############################
            start_time = time.time()
            if 'drl' in scheme:
                np.random.shuffle(unlabeled_nl_index)
                np.random.shuffle(unlabeled_cf_index)
            if 'dae' in scheme:
                np.random.shuffle(self_train_nl_index)
                np.random.shuffle(self_train_cf_index)
            if 'bt' in scheme:
                np.random.shuffle(pseudo_nl_index)
                np.random.shuffle(pseudo_cf_index)
            if labeled_data != []:
                np.random.shuffle(labeled_train_index)
            losses = { 'nl2cf': [], 'cf2nl': [] }
            self.model.train()
            for j in range(0, nsentences, batchSize):
                self.model.zero_grad()
                if 'drl' in scheme:
                    """ Dual Learning Cycle Starts from NL2CF Model """
                    ###################### Obtain minibatch data ######################
                    inputs, lens, raw_in = get_minibatch(unlabeled_nl, self.vocab['nl2cf'], task='unlabeled_nl2cf',
                        data_index=unlabeled_nl_index, index=j, batch_size=batchSize, device=self.device['nl2cf'])
                    ######################## Forward Model ##########################
                    nl2cf_loss, cf2nl_loss = self.model(inputs, lens, raw_in, start_from='nl2cf')
                    losses['nl2cf'].append(nl2cf_loss.item())
                    losses['cf2nl'].append(cf2nl_loss.item())
                    nl2cf_loss.backward()
                    cf2nl_loss.backward()
                    gc.collect()
                    torch.cuda.empty_cache()

                    """ Dual Learning Cycle Starts from CF2NL Model """
                    ###################### Obtain minibatch data ######################
                    inputs, lens, (raw_in, _, _) = get_minibatch(unlabeled_cf, self.vocab['cf2nl'], task='unlabeled_cf2nl',
                        data_index=unlabeled_cf_index, index=j, batch_size=batchSize, device=self.device['cf2nl'])
                    ########################### Forward Model ########################
                    nl2cf_loss, cf2nl_loss = self.model(inputs, lens, raw_in, start_from='cf2nl')
                    losses['nl2cf'].append(nl2cf_loss.item())
                    losses['cf2nl'].append(cf2nl_loss.item())
                    nl2cf_loss.backward()
                    cf2nl_loss.backward()
                    gc.collect()
                    torch.cuda.empty_cache()

                if 'bt' in scheme:
                    """ Pseudo training for NL2CF and CF2NL model | Back-translation """
                    ###################### Obtain minibatch data ######################
                    inputs, lens, outputs, out_lens, _ = get_minibatch(
                        nl_pseudo_samples, self.vocab['nl2cf'], task='nl2cf', data_index=pseudo_nl_index,
                        index=j, batch_size=batchSize, device=self.device['nl2cf'])
                    ############################ Forward Model ############################
                    batch_scores = self.model.nl2cf_model(inputs, lens, outputs[:, :-1])
                    batch_loss = self.loss_function['nl2cf'](batch_scores, outputs[:, 1:])
                    losses['nl2cf'].append(batch_loss.item())
                    batch_loss.backward()

                    ###################### Obtain minibatch data ######################
                    inputs, lens, outputs, out_lens, _ = get_minibatch(
                        cf_pseudo_samples, self.vocab['cf2nl'], task='cf2nl', data_index=pseudo_cf_index,
                        index=j, batch_size=batchSize, device=self.device['cf2nl'])
                    ############################ Forward Model ############################
                    batch_scores = self.model.cf2nl_model(inputs, lens, outputs[:, :-1])
                    batch_loss = self.loss_function['cf2nl'](batch_scores, outputs[:, 1:])
                    losses['cf2nl'].append(batch_loss.item())
                    batch_loss.backward()

                    gc.collect()
                    torch.cuda.empty_cache()

                if 'dae' in scheme:
                    """ Self-training Learning for NL2CF and CF2NL Model """
                    ###################### Obtain minibatch data ######################
                    inputs, lens, outputs, out_lens, _ = get_minibatch(
                        unlabeled_cf, self.vocab['nl2cf'], task='self_train_nl2cf', data_index=self_train_cf_index,
                        index=j, batch_size=batchSize, device=self.device['nl2cf'], train_dataset=unlabeled_nl)
                    ############################ Forward Model ############################
                    batch_scores = self.model.nl2cf_model(inputs, lens, outputs[:, :-1])
                    batch_loss = self.loss_function['nl2cf'](batch_scores, outputs[:, 1:])
                    losses['nl2cf'].append(batch_loss.item())
                    batch_loss.backward()

                    ###################### Obtain minibatch data ######################
                    inputs, lens, outputs, out_lens, _ = get_minibatch(
                        unlabeled_nl, self.vocab['cf2nl'], task='self_train_cf2nl', data_index=self_train_nl_index,
                        index=j, batch_size=batchSize, device=self.device['cf2nl'], train_dataset=unlabeled_cf)
                    ############################ Forward Model ############################
                    batch_scores = self.model.cf2nl_model(inputs, lens, outputs[:, :-1])
                    batch_loss = self.loss_function['cf2nl'](batch_scores, outputs[:, 1:])
                    losses['cf2nl'].append(batch_loss.item())
                    batch_loss.backward()

                    gc.collect()
                    torch.cuda.empty_cache()

                if labeled_data != []:
                    """ Supervised Learning for NL2CF and CF2NL Model """
                    ###################### Obtain minibatch data ######################
                    inputs, lens, outputs, out_lens, _ = get_minibatch(
                        labeled_data, self.vocab['nl2cf'], task='nl2cf', data_index=labeled_train_index,
                        index=j, batch_size=batchSize, device=self.device['nl2cf'])
                    ############################ Forward Model ############################
                    batch_scores = self.model.nl2cf_model(inputs, lens, outputs[:, :-1])
                    batch_loss = self.loss_function['nl2cf'](batch_scores, outputs[:, 1:])
                    losses['nl2cf'].append(batch_loss.item())
                    batch_loss.backward()

                    ###################### Obtain minibatch data ######################
                    inputs, lens, outputs, out_lens, _ = get_minibatch(
                        labeled_data, self.vocab['cf2nl'], task='cf2nl', data_index=labeled_train_index,
                        index=j, batch_size=batchSize, device=self.device['cf2nl'])
                    ############################ Forward Model ############################
                    batch_scores = self.model.cf2nl_model(inputs, lens, outputs[:, :-1])
                    batch_loss = self.loss_function['cf2nl'](batch_scores, outputs[:, 1:])
                    losses['cf2nl'].append(batch_loss.item())
                    batch_loss.backward()

                    gc.collect()
                    torch.cuda.empty_cache()

                self.model.pad_embedding_grad_zero()
                self.optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            nl2cf_epoch_loss = np.sum(losses['nl2cf'], axis=0)
            cf2nl_epoch_loss = np.sum(losses['cf2nl'], axis=0)
            self.best_result['nl2cf_losses'].append(nl2cf_epoch_loss)
            self.best_result['cf2nl_losses'].append(cf2nl_epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t NL2CF Loss: %.5f\t CF2NL Loss: %.5f' \
                                % (i, time.time() - start_time, nl2cf_epoch_loss, cf2nl_epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            ########################### Evaluation Phase ############################
            start_time = time.time()
            self_dev_nl_bleu, self_dev_cf_acc, self_dev_lf_acc = self.unsupervised_evaluation(dev_dataset,
                os.path.join(self.exp_path, 'self_valid.iter' + str(i)), test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Unsupervised Evaluation Dev:\tEpoch : %d\tTime : %.4fs\tNL2CF (NL Bleu : %.4f)\tCF2NL (CF/LF Acc : %.4f/%.4f)' \
                                % (i, time.time() - start_time, self_dev_nl_bleu, self_dev_cf_acc, self_dev_lf_acc))
            start_time = time.time()
            dev_cf_acc, dev_lf_acc, dev_nl_bleu  = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Evaluation Dev:\tEpoch : %d\tTime : %.4fs\tNL2CF (CF/LF Acc : %.4f/%.4f)\tCF2NL (NL Bleu : %.4f)' \
                                % (i, time.time() - start_time, dev_cf_acc, dev_lf_acc, dev_nl_bleu))
            start_time = time.time()
            test_cf_acc, test_lf_acc, test_nl_bleu = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Evaluation Test:\tEpoch : %d\tTime : %.4fs\tNL2CF (CF/LF Acc : %.4f/%.4f)\tCF2NL (NL Bleu : %.4f)' \
                                % (i, time.time() - start_time, test_cf_acc, test_lf_acc, test_nl_bleu))

            ######################## Pick best result and save #####################
            if self_dev_nl_bleu + self_dev_cf_acc >  self.best_result['self_dev_nl_bleu'] + self.best_result['self_dev_cf_acc']:
            # if dev_cf_acc + dev_nl_bleu > self.best_result['dev_cf_acc'] + self.best_result['dev_nl_bleu']:
                self.best_result['iter'] = i
                self.best_result['self_dev_nl_bleu'], self.best_result['self_dev_cf_acc'], self.best_result['self_dev_lf_acc'] = self_dev_nl_bleu, self_dev_cf_acc, self_dev_lf_acc
                self.logger.info('NEW BEST Unsupervised Metric:\tEpoch : %d\tBest Valid (NL Bleu : %.4f ; CF/LF Acc : %.4f/%.4f)'
                    % (i, self_dev_nl_bleu, self_dev_cf_acc, self_dev_lf_acc))
                self.model.save_model(nl2cf_save_dir=os.path.join(self.exp_path, 'nl2cf_model.pkl'))
                self.best_result['dev_cf_acc'], self.best_result['test_cf_acc'] = dev_cf_acc, test_cf_acc
                self.best_result['dev_lf_acc'], self.best_result['test_lf_acc'] = dev_lf_acc, test_lf_acc
                self.logger.info('NEW BEST NL2CF:\tBest Valid (CF/LF Acc : %.4f/%.4f)\tBest Test (CF/LF Acc : %.4f/%.4f)' \
                    % (dev_cf_acc, dev_lf_acc, test_cf_acc, test_lf_acc))
                self.model.save_model(cf2nl_save_dir=os.path.join(self.exp_path, 'cf2nl_model.pkl'))
                self.best_result['dev_nl_bleu'], self.best_result['test_nl_bleu'] = dev_nl_bleu, test_nl_bleu
                self.logger.info('NEW BEST CF2NL:\tBest Valid (NL Bleu : %.4f)\tBest Test (NL Bleu : %.4f)' 
                    % (dev_nl_bleu, test_nl_bleu))

            gc.collect()
            torch.cuda.empty_cache()

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST Unsupervised RESULT:\tEpoch : %d\tBest Valid (NL Bleu : %.4f ; CF/LF Acc : %.4f/%.4f)'
                % (self.best_result['iter'], self.best_result['self_dev_nl_bleu'], self.best_result['self_dev_cf_acc'], self.best_result['self_dev_lf_acc']))
        self.logger.info('NL2CF FINAL BEST RESULT:\tBest Valid (CF/LF Acc : %.4f/%.4f)\tBest Test (CF/LF Acc : %.4f/%.4f)'
                % (self.best_result['dev_cf_acc'], self.best_result['dev_lf_acc'], self.best_result['test_cf_acc'], self.best_result['test_lf_acc']))
        self.logger.info('CF2NL FINAL BEST RESULT:\tBest Valid (NL Bleu : %.4f)\tBest Test (NL Bleu : %.4f)'
                % (self.best_result['dev_nl_bleu'], self.best_result['test_nl_bleu']))
        self.model.load_model(os.path.join(self.exp_path, 'nl2cf_model.pkl'), os.path.join(self.exp_path, 'cf2nl_model.pkl'))
        json.dump(self.best_result, open(os.path.join(self.exp_path, 'history.json'), 'w'), indent=4)

#coding=utf8
import time, os, gc, json
from utils.example import Example
from utils.batch import get_minibatch, get_minibatch_sp, get_minibatch_unlabeled_cf2nl, get_minibatch_unlabeled_nl2cf
import numpy as np
import torch

class FSSPSolver():

    def __init__(self, nl2cf_model, cf2nl_model, sp_model, vocab, loss_function, optimizer, exp_path, logger, device, method='wmd', **kargs):
        super(FSSPSolver, self).__init__()
        self.nl2cf_model, self.cf2nl_model, self.sp_model = nl2cf_model, cf2nl_model, sp_model
        self.vocab = vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
        self.method = method
        if self.method == 'wmd':
            self.noise = Example.noise
        self.best_result = {
            "nl2cf_losses": [], "cf2nl_losses": [], "nl2cf_iter": 0, "cf2nl_iter": 0,
            "dev_cf_acc": 0., "dev_lf_acc": 0., "dev_nl_bleu": 0.,
            "test_cf_acc": 0., "test_lf_acc": 0., "test_nl_bleu": 0. }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, total_cf_acc, total_lf_acc, total_nl_bleu = len(data_index), [], [], []
        domain = Example.domain
        self.nl2cf_model.eval()
        self.cf2nl_model.eval()
        self.sp_model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, _, (raw_cf_inputs, raw_cf_outputs, raw_lf_outputs, variables) = get_minibatch(
                    data_inputs, self.vocab['nl2cf'], task='nl2cf', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device['nl2cf'])
                ############################ Forward NL2CF Model ############################
                with torch.no_grad():
                    results = self.nl2cf_model.decode_batch(inputs, lens, self.vocab['nl2cf'].cf2id, beam_size=beam, n_best=n_best)
                    cf_predictions = results["predictions"]
                    cf_predictions = [each[0] for each in cf_predictions]
                    cf_predictions = domain.reverse(cf_predictions, self.vocab['nl2cf'].id2cf)
                cf_acc = domain.compare_cf(cf_predictions, raw_cf_outputs)
                total_cf_acc.extend(cf_acc)
                tmp_ex_list = [Example(need=['cf', 'lf'], cf=' '.join(each), lf=' '.join(raw_lf_outputs[idx]), variables=variables[idx]) for idx, each in enumerate(cf_predictions)]
                ############################ Forward Semantic Parsing Model ############################
                inputs, lens, outputs, _, _ = get_minibatch_sp(tmp_ex_list, self.vocab['sp'], device=self.device['sp'], cf_vocab=True, cf_input=True)
                with torch.no_grad():
                    results = self.sp_model.decode_batch(inputs, lens, self.vocab['sp'].lf2id, beam_size=beam, n_best=n_best)
                    lf_predictions = results["predictions"]
                    lf_predictions = [pred for each in lf_predictions for pred in each]
                    lf_predictions = domain.reverse(lf_predictions, self.vocab['sp'].id2lf)
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
                    data_inputs, self.vocab['cf2nl'], task='cf2nl', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device['cf2nl'])
                ############################ Forward CF2NL Model ############################
                with torch.no_grad():
                    results = self.cf2nl_model.decode_batch(inputs, lens, self.vocab['cf2nl'].nl2id, beam_size=beam, n_best=n_best)
                    nl_predictions = results["predictions"]
                    nl_predictions = [each[0] for each in nl_predictions]
                    nl_predictions = domain.reverse(nl_predictions, self.vocab['cf2nl'].id2nl)
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

    def train_and_decode(self, unlabeled_nl, unlabeled_cf, dev_dataset, test_dataset, batchSize=16, test_batchSize=128,
            max_epoch=100, beam=5, n_best=1):
        faked_nl_samples = self.generate_faked_samples(unlabeled_nl, side='nl')
        faked_cf_samples = self.generate_faked_samples(unlabeled_cf, side='cf')
        faked_samples = faked_nl_samples + faked_cf_samples
        train_index = np.arange(len(faked_samples))
        nsentences = len(faked_samples)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_index)
            losses = {'nl2cf': [], 'cf2nl': []}
            self.nl2cf_model.train()
            self.cf2nl_model.train()
            for j in range(0, nsentences, batchSize):
                self.nl2cf_model.zero_grad()
                self.cf2nl_model.zero_grad()
                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, out_lens, _ = get_minibatch(
                    faked_samples, self.vocab['nl2cf'], task='nl2cf', data_index=train_index,
                    index=j, batch_size=batchSize, device=self.device['nl2cf'])
                ############################ Forward Model ############################
                batch_scores = self.nl2cf_model(inputs, lens, outputs[:, :-1])
                batch_loss = self.loss_function["nl2cf"](batch_scores, outputs[:, 1:])
                losses['nl2cf'].append(batch_loss.item())
                batch_loss.backward()

                ###################### Obtain minibatch data ######################
                inputs, lens, outputs, out_lens, _ = get_minibatch(
                    faked_samples, self.vocab['cf2nl'], task='cf2nl', data_index=train_index,
                    index=j, batch_size=batchSize, device=self.device['cf2nl'])
                ############################ Forward Model ############################
                batch_scores = self.cf2nl_model(inputs, lens, outputs[:, :-1])
                batch_loss = self.loss_function["cf2nl"](batch_scores, outputs[:, 1:])
                losses['cf2nl'].append(batch_loss.item())
                batch_loss.backward()

                self.nl2cf_model.pad_embedding_grad_zero()
                self.cf2nl_model.pad_embedding_grad_zero()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            nl2cf_epoch_loss, cf2nl_epoch_loss = np.sum(losses['nl2cf']), np.sum(losses['cf2nl'])
            self.best_result['nl2cf_losses'].append(nl2cf_epoch_loss)
            self.best_result['cf2nl_losses'].append(cf2nl_epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t NL2CF Loss: %.5f\t CF2NL Loss: %.5f' \
                                % (i, time.time() - start_time, nl2cf_epoch_loss, cf2nl_epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_cf_acc, dev_lf_acc, dev_nl_bleu = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Evaluation Dev:\tEpoch : %d\tTime : %.4fs\tNL2CF (CF/LF Acc : %.4f/%.4f)\tCF2NL (NL Bleu : %.4f)' \
                                % (i, time.time() - start_time, dev_cf_acc, dev_lf_acc, dev_nl_bleu))
            start_time = time.time()
            test_cf_acc, test_lf_acc, test_nl_bleu = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Evaluation Test:\tEpoch : %d\tTime : %.4fs\tNL2CF (CF/LF Acc : %.4f/%.4f)\tCF2NL (NL Bleu : %.4f)' \
                                % (i, time.time() - start_time, test_cf_acc, test_lf_acc, test_nl_bleu))

            ######################## Pick best result on dev and save #####################
            if dev_cf_acc > self.best_result['dev_cf_acc']:
                self.best_result['nl2cf_iter'] = i
                self.nl2cf_model.save_model(os.path.join(self.exp_path, 'nl2cf_model.pkl'))
                self.best_result['dev_cf_acc'], self.best_result['test_cf_acc'] = dev_cf_acc, test_cf_acc
                self.best_result['dev_lf_acc'], self.best_result['test_lf_acc'] = dev_lf_acc, test_lf_acc
                self.logger.info('NL2CF NEW BEST:\tEpoch: %d;\tBest Valid Acc (CF/LF) : %.4f/%.4f;\tBest Test Acc (CF/LF) : %.4f/%.4f' % (i, dev_cf_acc, dev_lf_acc, test_cf_acc, test_lf_acc))
            if dev_nl_bleu > self.best_result['dev_nl_bleu']:
                self.best_result['cf2nl_iter'] = i
                self.cf2nl_model.save_model(os.path.join(self.exp_path, 'cf2nl_model.pkl'))
                self.best_result['dev_nl_bleu'], self.best_result['test_nl_bleu'] = dev_nl_bleu, test_nl_bleu
                self.logger.info('CF2NL NEW BEST:\tEpoch: %d;\tBest Valid Bleu : %.4f;\tBest Test Bleu : %.4f' % (i, dev_nl_bleu, test_nl_bleu))

        ######################## Reload best model for later usage #####################
        self.logger.info('NL2CF FINAL BEST RESULT:\tEpoch: %d;\tBest Valid (CF/LF Acc : %.4f/%.4f)\tBest Test (CF/LF Acc : %.4f/%.4f)'
                % (self.best_result['nl2cf_iter'], self.best_result['dev_cf_acc'], self.best_result['dev_lf_acc'], self.best_result['test_cf_acc'], self.best_result['test_lf_acc']))
        self.logger.info('CF2NL FINAL BEST RESULT:\tEpoch: %d;\tBest Valid (NL Bleu : %.4f)\tBest Test (NL Bleu : %.4f)'
                % (self.best_result['cf2nl_iter'], self.best_result['dev_nl_bleu'], self.best_result['test_nl_bleu']))
        self.nl2cf_model.load_model(os.path.join(self.exp_path, 'nl2cf_model.pkl'))
        self.cf2nl_model.load_model(os.path.join(self.exp_path, 'cf2nl_model.pkl'))
        json.dump(self.best_result, open(os.path.join(self.exp_path, 'history.json'), 'w'), indent=4)

    def generate_faked_samples(self, train_dataset, side='nl'):

        def max_overlapping(sent, all_candidates):
            overlap = [len(set(sent) & set(ex)) for ex in all_candidates]
            union = [len(set(sent) | set(ex)) for ex in all_candidates]
            score = [i / float(j) if j != 0. else 0. for i, j in zip(overlap, union)]
            return all_candidates[score.index(max(score))]

        start_time = time.time()
        candidates = [ex.cf if side == 'nl' else ex.nl for ex in train_dataset]
        if self.method == 'wmd':
            faked_samples = [
                (ex.nl, self.noise.pick_candidate(ex.nl, candidates)[0]) if side == 'nl' \
                else (self.noise.pick_candidate(ex.cf, candidates)[0], ex.cf)
                for ex in train_dataset ]
        elif self.method == 'bow':
            faked_samples = [
                (ex.nl, max_overlapping(ex.nl, candidates)) if side == 'nl' \
                else (max_overlapping(ex.cf, candidates), ex.cf)
                for ex in train_dataset ]
        else:
            raise ValueError('Unknown faked sample method !')
        faked_samples = [ Example(need=['nl', 'cf'], nl=' '.join(nl), cf=' '.join(cf)) for nl, cf in faked_samples ]
        self.logger.info('It takes %.4fs to create %d faked samples ...' % (time.time() - start_time, len(faked_samples)))
        return faked_samples

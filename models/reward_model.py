#coding=utf8
import os, json, torch
from utils.example import Example, UtteranceExample
from utils.domain import get_bleu_score
from utils.batch import get_minibatch_language_model, get_minibatch_text_style_classification, get_minibatch_semantic_parsing
from models.model_utils import lens2mask
from models.model_constructor import construct_model


class RewardModel(): # do not inherit from nn.Module, gradient is not calculated

    def __init__(self, language_model_path, tsc_model_path, nsp_model_path, device='cpu', reward_type="flu+sty+rel"):
        super(RewardModel, self).__init__()
        lm_params = json.load(open(os.path.join(language_model_path, 'params.json'), 'r'))
        self.language_model = construct_model['language_model'](**lm_params).to(device)
        check_point = torch.load(open(os.path.join(language_model_path, 'model.pkl'), 'rb'), map_location=device)
        self.language_model.load_state_dict(check_point['model'])
        self.language_model.eval()

        tsc_params = json.load(open(os.path.join(tsc_model_path, 'params.json'), 'r'))
        self.text_style_classifier = construct_model['text_style_classification'](**tsc_params).to(device)
        check_point = torch.load(open(os.path.join(tsc_model_path, 'model.pkl'), 'rb'), map_location=device)
        self.text_style_classifier.load_state_dict(check_point['model'])
        self.text_style_classifier.eval()

        nsp_params = json.load(open(os.path.join(nsp_model_path, 'params.json'), 'r'))
        self.nsp_model = construct_model['semantic_parsing'](**nsp_params).to(device)
        check_point = torch.load(open(os.path.join(nsp_model_path, 'model.pkl'), 'rb'), map_location=device)
        self.nsp_model.load_state_dict(check_point['model'])
        self.nsp_model.eval()

        self.device = device
        self.reward_type = reward_type


    def forward(self, *args, choice='nl2cf2nl_val'):
        if choice == 'nl2cf2nl_val':
            return self.nl2cf2nl_validity_reward(*args)
        elif choice == 'cf2nl2cf_val':
            return self.cf2nl2cf_validity_reward(*args)
        # elif choice == 'nl2cf2nl_rec':
        #     return self.nl2cf2nl_reconstruction_reward(*args)
        # elif choice == 'cf2nl2cf_rec':
        #     return self.cf2nl2cf_reconstruction_reward(*args)
        else: # loglikelihood is better than bleu score or utterance accuracy
            return self.reconstruction_reward(*args)


    def nl2cf2nl_validity_reward(self, cf_list, variables=[]):
        val_reward = torch.zeros(len(cf_list), dtype=torch.float, device='cpu')
        if 'flu' in self.reward_type:
            # calculate canonical form language model length normalized log probability
            ex_list = [Example(cf=cf) for cf in cf_list]
            inputs, lens = get_minibatch_language_model(ex_list, self.device, input_side='cf')
            with torch.no_grad():
                logprob = self.language_model.sentence_logprob(inputs, lens, input_side='cf').cpu()
            val_reward += logprob
            # calculate LF execution reward, 0/1 indicator
            inputs, lens = get_minibatch_semantic_parsing(ex_list, self.device, input_side='cf', labeled=False)
            domain, vocab = Example.domain, Example.vocab
            with torch.no_grad():
                predictions = self.nsp_model.decode_batch(inputs, lens, vocab.lf2id, beam_size=5, n_best=1)['predictions']
                predictions = domain.reverse(sum(predictions, []), vocab.id2lf)
                pred_ans = domain.obtain_denotations(domain.normalize(predictions, variables))
                grammar_check = torch.tensor(domain.is_valid(pred_ans), dtype=torch.float, device='cpu')
            val_reward += grammar_check
        if 'sty' in self.reward_type:
            # calculate text style reward
            ex_list = [UtteranceExample(nl=cf) for cf in cf_list]
            inputs = get_minibatch_text_style_classification(ex_list, self.device, labeled=False)
            with torch.no_grad():
                results = self.text_style_classifier(inputs).cpu()
            val_reward += results # label is 1 if input is canonical utterance
        return val_reward


    def cf2nl2cf_validity_reward(self, nl_list, vairables=[]):
        val_reward = torch.zeros(len(nl_list), dtype=torch.float, device='cpu')
        if 'flu' in self.reward_type:
            # calculate natural language model length normalized log probability
            ex_list = [Example(nl=nl) for nl in nl_list]
            inputs, lens = get_minibatch_language_model(ex_list, self.device, input_side='nl')
            with torch.no_grad():
                logprob = self.language_model.sentence_logprob(inputs, lens, input_side='nl').cpu()
            val_reward += logprob
        if 'sty' in self.reward_type:
            # calculate text style reward
            ex_list = [UtteranceExample(nl=nl) for nl in nl_list]
            inputs = get_minibatch_text_style_classification(ex_list, self.device, labeled=False)
            with torch.no_grad():
                results = self.text_style_classifier(inputs).cpu()
            val_reward += (1 - results) # label is 0 if input is natural language utterance
        return val_reward


    def reconstruction_reward(self, logscores, references, lens):
        """
        @args:
            logscores: bsize x max_out_len - 1 x vocab_size
            references: bsize x max_out_len
            lens: len for each sample
        """
        references, lens = references[:, 1:], lens - 1
        mask = lens2mask(lens)
        pick_score = torch.gather(logscores, dim=-1, index=references.unsqueeze(dim=-1)).squeeze(dim=-1)
        masked_score = mask.float() * pick_score # remove PAD scores
        reward = masked_score.sum(dim=1)
        return reward


    def nl2cf2nl_reconstruction_reward(self, predictions, references):
        scores = list(map(lambda x: get_bleu_score(x[0], [x[1]]), zip(predictions, references)))
        return torch.tensor(scores, dtype=torch.float)


    def cf2nl2cf_reconstruction_reward(self, predictions, references):
        scores = list(map(lambda x: get_bleu_score(x[0], [x[1]], weights=(0, 0, 0.5, 0.5)), zip(predictions, references)))
        return torch.tensor(scores, dtype=torch.float)


    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)
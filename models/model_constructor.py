#coding=utf8
from models.semantic_parser import semantic_parsing_model, multitask_semantic_parsing_model
from models.language_model import DualLanguageModel
from models.paraphrase_model import ParaphraseModel, dual_paraphrase_model
from models.text_style_classifier import TextStyleClassifier
from models.cycle_learning import CycleLearningModel


construct_model = {
    "semantic_parsing": semantic_parsing_model,
    "multitask_semantic_parsing": multitask_semantic_parsing_model, # used in one-stage semantic parsing model
    "paraphrase": ParaphraseModel,
    "dual_paraphrase": dual_paraphrase_model,
    "cycle_learning": CycleLearningModel,
    "language_model": DualLanguageModel,
    "text_style_classification": TextStyleClassifier
}
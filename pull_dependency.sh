#!/bin/bash

mkdir evaluator lib pretrained_models

echo "Start downloading evaluator for OVERNIGHT and GEO datasets ..."
wget -c https://worksheets.codalab.org/rest/bundles/0xbfbf0d1d8ab94874a68646a7d66c478e/contents/blob/ -O evaluator.tar.gz
echo "Start downloading libraries for evaluation ..."
wget -c https://worksheets.codalab.org/rest/bundles/0xc6821b4f13f445d1b54e9da63019da1d/contents/blob/ -O lib.tar.gz
tar -zxf evaluator.tar.gz -C evaluator
tar -zxf lib.tar.gz -C lib
rm -rf evaluator.tar.gz lib.tar.gz
mkdir -p lib/data/overnight
mv data/geo/geo880* lib/data/overnight/
cp evaluator/sempre/module-classes.txt .

echo "Start downloading pre-trained models, including GloVe/GoogleNews word vectors, ELMo and BERT ..."
wget -c http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d pretrained_models/
rm glove.6B.zip

wget -c -P pretrained_models https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
wget -c -P pretrained_models https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget -c -P pretrained_models https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

git lfs install
git clone https://huggingface.co/bert-base-uncased pretrained_models/bert-base-uncased
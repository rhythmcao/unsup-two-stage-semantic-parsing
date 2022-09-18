#!/bin/bash

task='natural_language_generation'
dataset=$1
# read_model_path=''

# model paras
emb_size=100
hidden_dim=200
num_layers=1
cell=lstm # lstm, gru
trans=empty # empty, tanh(affine)

# training paras
lr=0.001
l2=1e-5
dropout=0.5
batchSize=16
test_batchSize=128
init_weight=0.2
max_norm=5
max_epoch=100
beam=5
n_best=1

# special paras
labeled=1.0
deviceId=0
seed=999

python3 scripts/natural_language_generation.py --task $task --emb_size $emb_size --hidden_dim $hidden_dim --num_layers $num_layers \
    --trans $trans --dataset $dataset --cell $cell --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --test_batchSize $test_batchSize \
    --init_weight $init_weight --max_norm $max_norm --max_epoch $max_epoch --beam $beam --n_best $n_best \
    --labeled $labeled --deviceId $deviceId --seed $seed

#!/bin/bash

task=faked_sample_two_stage_${2}
dataset=$1
read_nsp_model_path=exp/task_naive_semantic_parsing/dataset_${1}__labeled_1.0/cell_lstm__emb_100__hidden_200_x_1__trans_empty__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100__beam_5__nbest_1/
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
test_batchSize=64
init_weight=0.2
max_norm=5
max_epoch=100
beam=5
n_best=1

# special paras
method=$2 # wmd, bow
shared_encoder='--shared_encoder'
labeled=1.0
deviceId=0
seed=999

python3 scripts/faked_sample_two_stage.py --task $task --emb_size $emb_size --hidden_dim $hidden_dim --num_layers $num_layers --read_nsp_model_path $read_nsp_model_path \
    --trans $trans --dataset $dataset --cell $cell --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --test_batchSize $test_batchSize \
    --init_weight $init_weight --max_norm $max_norm --max_epoch $max_epoch --beam $beam --n_best $n_best \
    --labeled $labeled --deviceId $deviceId --seed $seed --method $method $shared_encoder

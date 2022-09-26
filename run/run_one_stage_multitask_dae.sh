#!/bin/bash
task=one_stage_multitask_dae
dataset=$1
# testing='--testing'
# read_model_path=path_to_model

# model params
embed_size=100
hidden_size=200
num_layers=1
cell=lstm

# training params
lr=0.001
l2=1e-4
dropout=0.5
batch_size=16
test_batch_size=64
init_weight=0.2
max_norm=5
eval_after_epoch=60
max_epoch=100
beam_size=5
n_best=1

# special parasm
noise_type=drop+addition+shuffling
deviceId=0
seed=999

python3 scripts/one_stage_multitask_dae.py --task $task --dataset $dataset --embed_size $embed_size --hidden_size $hidden_size \
    --cell $cell --num_layers $num_layers --lr $lr --l2 $l2 --dropout $dropout --batch_size $batch_size --test_batch_size $test_batch_size \
    --max_norm $max_norm --max_epoch $max_epoch --init_weight $init_weight --eval_after_epoch $eval_after_epoch \
    --noise_type $noise_type --beam_size $beam_size --n_best $n_best --deviceId $deviceId --seed $seed

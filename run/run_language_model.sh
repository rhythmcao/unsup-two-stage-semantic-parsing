#!/bin/bash
task=language_model
dataset=$1
# testing='--testing'
# read_model_path=''

# model params
embed_size=100
hidden_size=200
num_layers=1
cell=lstm # lstm, gru

# training params
lr=0.001
l2=1e-4
dropout=0.5
batch_size=16
test_batch_size=128
init_weight=0.2
max_norm=5
eval_after_epoch=60
max_epoch=100

# special params
deviceId=-1
seed=999

python scripts/language_model.py --task $task --dataset $dataset --embed_size $embed_size --hidden_size $hidden_size --share_encoder \
    --cell $cell --num_layers $num_layers --lr $lr --l2 $l2 --dropout $dropout --batch_size $batch_size --test_batch_size $test_batch_size \
    --init_weight $init_weight --max_norm $max_norm --eval_after_epoch $eval_after_epoch --max_epoch $max_epoch --deviceId $deviceId --seed $seed

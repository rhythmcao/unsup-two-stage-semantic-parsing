#!/bin/bash
task=pretrained_embed
dataset=$1
# testing='--testing'
# read_model_path=path_to_model

# model params
pretrained_embed=$2
embed_size=100
hidden_size=200
num_layers=1
cell=lstm

# training params
if [ "${pretrained_embed}" = "bert" ] ; then
    lr=0.001
    l2=1e-4
    layerwise_decay=0.
    lr_schedule=constant
elif [ "${pretrained_embed}" = "elmo" ] ; then
    lr=0.001
    l2=1e-4
    layerwise_decay=1.
    lr_schedule=constant
else
    lr=0.001
    l2=1e-4
    layerwise_decay=1.
    lr_schedule=constant
fi

dropout=0.5
batch_size=16
test_batch_size=64
init_weight=0.2
max_norm=5
eval_after_epoch=60
max_epoch=100
beam_size=5
n_best=1

# special params
train_input_side=cf
eval_input_side=nl
deviceId=0
seed=999

python3 scripts/one_stage_semantic_parsing.py --task $task --dataset $dataset --pretrained_embed $pretrained_embed --embed_size $embed_size \
    --hidden_size $hidden_size --cell $cell --num_layers $num_layers --lr $lr --l2 $l2 --batch_size $batch_size --test_batch_size $test_batch_size \
    --dropout $dropout --layerwise_decay $layerwise_decay --lr_schedule $lr_schedule --max_norm $max_norm --init_weight $init_weight \
    --eval_after_epoch $eval_after_epoch --max_epoch $max_epoch --beam_size $beam_size --n_best $n_best \
    --train_input_side $train_input_side --eval_input_side $eval_input_side --deviceId $deviceId --seed $seed

#!/bin/bash
task='discriminator'
dataset=$1
# read_model_path=''

emb_size=100
filters="3 4 5"
filters_num="10 20 30"

batchSize=50
test_batchSize=128
lr=1
l2=1e-5
reduction=sum
init_weight=0.2
dropout=0.5
max_norm=5
max_epoch=100
labeled=1.0
deviceId=0
seed=999

python scripts/discriminator.py --task $task --dataset $dataset --emb_size $emb_size --filters $filters --filters_num $filters_num \
    --reduction $reduction --batchSize $batchSize --test_batchSize $test_batchSize --lr $lr --l2 $l2 --dropout $dropout --max_norm $max_norm \
    --init_weight $init_weight --seed $seed --labeled $labeled --max_epoch $max_epoch --deviceId $deviceId

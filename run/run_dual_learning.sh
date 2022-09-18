#!/bin/bash

task=dual_learning_nl_$2
dataset=$1

read_pretrained_model_path=exp/task_pretrain_nl2cf_and_cf2nl_nl_${2}/dataset_${1}__labeled_0.0/cell_lstm__emb_100__hidden_200_x_1__trans_empty__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_50__beam_5__nbest_1__noisy_drop+add+shuffle_shared/
read_lm_path=exp/task_language_model/dataset_${1}__labeled_1.0/cell_lstm__emb_100__hidden_200_x_1__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100/
read_nsp_model_path=exp/task_naive_semantic_parsing/dataset_${1}__labeled_1.0/cell_lstm__emb_100__hidden_200_x_1__trans_empty__dropout_0.5__reduce_sum__lr_0.001__mn_5.0__l2_1e-05__bsize_16__me_100__beam_5__nbest_1/
read_sty_path=exp/task_discriminator/dataset_${1}__labeled_1.0/emb_100__filter_3x10_4x20_5x30__dropout_0.5__reduce_sum__lr_1.0__mn_5.0__l2_1e-05__bsize_50__me_100/
# read_model_path=

# training paras
reduction=sum # sum, mean
lr=0.001
l2=1e-5
batchSize=16
test_batchSize=64
max_norm=5
max_epoch=50
beam=5
n_best=1

# special paras
sample=6
alpha=0.5
beta=0.5
labeled=0.0
nl_labeled=$2
scheme=bt+drl
reward=flu+sty+rel
deviceId="0 1"
seed=999

python3 scripts/dual_learning.py --task $task --dataset $dataset \
    --read_pretrained_model_path $read_pretrained_model_path --read_nsp_model_path $read_nsp_model_path --read_lm_path $read_lm_path --read_sty_path $read_sty_path \
    --reduction $reduction --lr $lr --l2 $l2 --batchSize $batchSize --test_batchSize $test_batchSize \
    --scheme $scheme --max_norm $max_norm --max_epoch $max_epoch --beam $beam --n_best $n_best --sample $sample --alpha $alpha --beta $beta \
    --nl_labeled $nl_labeled --reward $reward --labeled $labeled --deviceId $deviceId --seed $seed #--testing --read_model_path $read_model_path

#!/bin/bash
task=cycle_learning
dataset=$1
# testing='--testing'
# read_model_path=

read_pdp_model_path=exp/task_two_stage_multitask_dae/dataset_${1}__noise_drop+addition+shuffling__enc_shared__cell_lstm__es_100__hd_200_x_1__dp_0.5__lr_0.001__l2_0.0001__mn_5.0__bs_16__me_50__bm_5__nb_1__seed_999
read_nsp_model_path=exp/task_naive_semantic_parsing/dataset_${1}__cell_lstm__es_100__hd_200_x_1__dp_0.5__lr_0.001__l2_0.0001__ld_1.0__sd_constant__mn_5.0__bs_16__me_100__bm_5__nb_1__seed_999
read_language_model_path=exp/task_language_model/dataset_${1}__cell_lstm__es_100__hd_200_x_1__dp_0.5__lr_0.001__l2_0.0001__mn_5.0__bs_16__me_100__seed_999
read_tsc_model_path=exp/task_text_style_classification/dataset_${1}__es_100__ft_3x10+4x20+5x30__dp_0.5__lr_1.0__l2_0.0001__mn_5.0__bs_50__me_50__seed_999

# training params
lr=0.001
l2=1e-4
lr_schedule=constant
batch_size=16
test_batch_size=64
max_norm=5
max_epoch=50
beam_size=5
n_best=1

# special params
labeled=$2
alpha=0.5
beta=0.5
sample_size=6
train_scheme=dbt+drl
reward_type=flu+sty+rel
noise_type=drop+addition+shuffling
deviceId=0
seed=999

python3 scripts/cycle_learning.py --task $task --dataset $dataset --read_pdp_model_path $read_pdp_model_path \
    --read_nsp_model_path $read_nsp_model_path --read_language_model_path $read_language_model_path --read_tsc_model_path $read_tsc_model_path \
    --lr $lr --l2 $l2 --batch_size $batch_size --test_batch_size $test_batch_size --max_norm $max_norm --lr_schedule $lr_schedule \
    --train_scheme $train_scheme --beam_size $beam_size --n_best $n_best --sample_size $sample_size --alpha $alpha --beta $beta \
    --max_epoch $max_epoch --reward_type $reward_type --noise_type $noise_type --labeled $labeled --deviceId $deviceId --seed $seed

#!/bin/bash
#model_size=$1
#echo $model_size
#model_dir=$2
#profile_dir=$3
export HIP_VISIBLE_DEVICES=4
rocprof --stats python3 gpt2_1step.py tf --learning_rate 5e-5 --output_dir . $*
#python3 gpt2_profile.py $profile_dir
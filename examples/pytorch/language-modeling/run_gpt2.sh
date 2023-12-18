#!/bin/bash
# MI300 Envs
#export HSA_DISABLE_CACHE=1

### F8 specific envs
#export F8_CONFIRM=0
#export F8_SIM=0
##
##
### Limit GPUs
###export HIP_VISIBLE_DEVICES=0
##
##
### rocBLAS output
export ROCBLAS_LAYER=2
##
### rocBLAS numerical checking
###export ROCBLAS_CHECK_NUMERICS=2
###export ROCBLAS_LAYER=7
##
###MIOpen numerical checking
###export MIOPEN_CHECK_NUMERICS=9
###export MIOPEN_CHECK_NUMERICS=1
###export MIOPEN_DUMP_TENSOR_PATH=/home/examples/imagenet/abnorm/abnorm_tens
##export MIOPEN_DISABLE_CACHE=1
##
###MIOpen logging
#export MIOPEN_ENABLE_LOGGING=1
###export MIOPEN_ENABLE_LOGGING_MPMT=0
##export MIOPEN_ENABLE_LOGGING_CMD=1
#export MIOPEN_LOG_LEVEL=7

N=1

export ENABLE_F8_GEMM=0
export ENABLE_F8_CONV=0
for i in $(seq 1 $N);
do
	python3 run_clm.py --model_name_or_path gpt2 \
			   --dataset_name wikitext \
			   --dataset_config_name wikitext-103-raw-v1 \
			   --do_train \
			   --label_smoothing 0.1 \
			   --save_total_limit 5 \
			   --dataloader_num_workers 1 \
			   --max_steps 100 \
			   --logging_steps 10 \
			   --output_dir /tmp/test-gpt2-fp16-$i \
			   --overwrite_output_dir \
			   --per_device_train_batch_size 24 \
			   --fp16 --half_precision_backend apex
done
echo "Done!"

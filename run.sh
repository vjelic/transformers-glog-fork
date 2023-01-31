#!/bin/bash

NGPUS=$(rocm-smi --showgpu | grep -Eo "GPU\[[0-9]*\]"  | sed -e 's/GPU\[//g' -e 's/\]//g' | wc -l)

USEGPUS=${1:-${NGPUS}}
MODEL=${2:-"bert"}
ddp_method=${3:-"torchdpp"}
deepspeed_conf=$4

#MASTER_NODE=`echo $SLURM_NODELIST | sed -e 's/\[\([0-9]\).*\]/\1/g' | sed -e 's/,.*//g'`

#MASTER_NODE=${MASTER_NODE:-`hostname`}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_NTASKS=${SLURM_NTASKS:-1}

HF_PATH=${HF_PATH:-`pwd`}
HF_HOME=${HF_HOME:-"${HF_PATH}/nas_share"}
batch_size=${batch_size:-24}

source_prefix_enable=false

CMD=""

if echo "bert" | grep -i "^${MODEL}$" > /dev/null ; then
    CMD+=" ${HF_PATH}/examples/pytorch/language-modeling/run_mlm.py"
    CMD+=" --model_name_or_path bert-large-uncased --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --output_dir /tmp/test-mlm-bbu"
elif echo "bart" | grep -i "^${MODEL}$" > /dev/null ; then
    CMD+=" $HF_PATH/examples/pytorch/translation/run_translation.py"
    CMD+=" --model_name_or_path facebook/bart-large --dataset_name wmt16 --dataset_config ro-en --output_dir /tmp/test-mlm-bbu"
    CMD+=" --label_smoothing 0.1"
    CMD+=" --predict_with_generate --source_lang en --target_lang ro --warmup_steps 5"
    batch_size=60
elif echo "t5-large" | grep -i "^${MODEL}$" > /dev/null ; then
    CMD+=" $HF_PATH/examples/pytorch/translation/run_translation.py"
    CMD+=" --model_name_or_path t5-large --dataset_name wmt16 --dataset_config ro-en --output_dir /tmp/tst-translation"
    CMD+=" --label_smoothing 0.1"
    CMD+=" --predict_with_generate --source_lang en --target_lang ro --warmup_steps 5"
    source_prefix_enable=true
    batch_size=30
else
    "this model not support"
    exit
fi

CMD+=" --do_train --max_steps 150 --logging_steps 1 --overwrite_output_dir --per_device_train_batch_size ${batch_size} --fp16 --skip_memory_metrics=True"

if [ "$ddp_method" == "deepspeed" ]; then
    CMD+=" --deepspeed=${deepspeed_conf}"
fi

if $source_prefix_enable ; then
    echo "$CMD --source_prefix \"translate English to Romanian: \""
    python3 -m torch.distributed.launch --node_rank ${SLURM_NODEID} --nnodes ${SLURM_NTASKS} --master_addr ${MASTER_NODE} --master_port 23456 --nproc_per_node ${USEGPUS}${CMD} --source_prefix "translate English to Romanian: "
else
    echo $CMD
    python3 -m torch.distributed.launch --node_rank ${SLURM_NODEID} --nnodes ${SLURM_NTASKS} --master_addr ${MASTER_NODE} --master_port 23456 --nproc_per_node ${USEGPUS}${CMD}
fi

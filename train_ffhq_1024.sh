#!/bin/bash
set -e

export NOISY=1
export DEBUG=1

config="config-f" # StyleGAN 2
#config="config-a" # StyleGAN 1

data_dir=gs://dota-euw4a/datasets
dataset=ffhq
mirror=true
metrics=none

export TPU_HOST=10.255.128.2
export TPU_NAME=tpu-v3-512-euw4a-53
cores=512
export IMAGENET_TFRECORD_DATASET='gs://dota-euw4a/datasets/ffhq1024/ffhq1024-0*'
run_name=run76
export MODEL_DIR=gs://dota-euw4a/runs/${run_name}-ffhq-1024-mirror/
export BATCH_PER=2
export BATCH_SIZE=$(($BATCH_PER * $cores))
export SPATIAL_PARTITION_FACTOR=2
export RESOLUTION=1024
export LABEL_SIZE=0
export LABEL_BIAS=0
export IMAGENET_UNCONDITIONAL=1
#export LABEL_FILE=gs://arfa-euw4a/datasets/e621-cond/e621-cond-rxx.labels

set -x
#exec python3 run_training.py --num-gpus="${cores}" --data-dir="${data_dir}" --config="${config}" --dataset="${dataset}" --mirror-augment="${mirror}" --metrics="${metrics}" "$@"
while true; do
  timeout --signal=SIGKILL 19h python3 run_training.py --num-gpus="${cores}" --data-dir="${data_dir}" --config="${config}" --dataset="${dataset}" --mirror-augment="${mirror}" --metrics="${metrics}" "$@" 2>&1 | tee -a "${run_name}.txt"
  sleep 30
done

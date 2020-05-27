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

set -x

if [ -f ./.env ]
then
  source ./.env
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST="${TPU_HOST:-10.255.128.2}"
export TPU_NAME="${TPU_NAME:-tpu-v3-512-euw4a-53}"
cores="$(echo $TPU_NAME | sed 's/^tpu-v[23][-]\([0-9]*\).*$/\1/g')"
if [ -z "$cores" ]
then
  1>&2 echo "Failed to parse TPU core count from $TPU_NAME"
  exit 1
fi
export IMAGENET_TFRECORD_DATASET="${IMAGENET_TFRECORD_DATASET:-gs://dota-euw4a/datasets/ffhq1024/ffhq1024-0*}"
export RUN_NAME="${RUN_NAME:-run76-ffhq-1024-mirror}"
export MODEL_DIR="${MODEL_DIR:-gs://dota-euw4a/runs/${RUN_NAME}}"
export BATCH_PER="${BATCH_PER:-3}"
export BATCH_SIZE="${BATCH_SIZE:-$(($BATCH_PER * $cores))}"
export SPATIAL_PARTITION_FACTOR="${SPATIAL_PARTITION_FACTOR:-2}"
export RESOLUTION="${RESOLUTION:-1024}"
export LABEL_SIZE="${LABEL_SIZE:-0}"
export LABEL_BIAS="${LABEL_BIAS:-0}"
export IMAGENET_UNCONDITIONAL="${IMAGENET_UNCONDITIONAL:-1}"
#export LABEL_FILE="${LABEL_FILE:-gs://arfa-euw4a/datasets/e621-cond/e621-cond-rxx.labels}"
export ITERATIONS_PER_LOOP="${ITERATIONS_PER_LOOP:-256}"
export HOST_CALL_EVERY_N_STEPS="${HOST_CALL_EVERY_N_STEPS:-64}"

if [ ! -z "$DD_API_KEY" ]
then
  #export DATADOG_TRACE_DEBUG=true
  export DD_PROFILING_CAPTURE_PCT=20
  #export DD_LOGS_INJECTION=true
  #export DD_TRACE_ANALYTICS_ENABLED=true
  export DD_VERSION="stylegan2-${RUN_NAME}"
  export DD_SERVICE=stylegan2
  bin="pyddprofile"
else
  bin="python3"
fi

tmux-set-title "stylegan2 | ${TPU_NAME}:${TPU_HOST} | ${RUN_NAME} | ${MODEL_DIR}"

#exec "$bin" run_training.py --num-gpus="${cores}" --data-dir="${data_dir}" --config="${config}" --dataset="${dataset}" --mirror-augment="${mirror}" --metrics="${metrics}" "$@"
while true; do
  timeout --signal=SIGKILL 19h "$bin" run_training.py --num-gpus="${cores}" --data-dir="${data_dir}" --config="${config}" --dataset="${dataset}" --mirror-augment="${mirror}" --metrics="${metrics}" "$@" 2>&1 | tee -a "${RUN_NAME}.txt"
  sleep 30
done

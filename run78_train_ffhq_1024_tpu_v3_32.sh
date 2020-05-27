#!/bin/bash
set -ex
export TPU_NAME="${TPU_NAME:-tpu-v3-32-euw4a-0}"
export RUN_NAME="${RUN_NAME:-run77-ffhq-1024-mirror-tpu-v3-32}"
exec bash run76_train_ffhq_1024.sh

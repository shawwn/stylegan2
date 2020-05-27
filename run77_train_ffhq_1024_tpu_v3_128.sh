#!/bin/bash
set -ex
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-50}"
export RUN_NAME="${RUN_NAME:-run77-ffhq-1024-mirror-tpu-v3-128}"
exec bash run76_train_ffhq_1024.sh

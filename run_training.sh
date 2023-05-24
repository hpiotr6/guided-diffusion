#!/bin/bash
TRAIN_FLAGS="--iterations 300000
--anneal_lr True
--batch_size 256
--lr 3e-4
--save_interval 10000
--weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128
--classifier_attention_resolutions 32,16,8
--classifier_depth 2
--classifier_width 128
--classifier_pool attention
--classifier_resblock_updown True
--classifier_use_scale_shift_norm True"

python \
    -m debugpy --listen 5678 --wait-for-client \
    -m scripts.classifier_train \
    --data_dir datasets/GT-RAIN/GT-RAIN_train \
    --val_data_dir datasets/GT-RAIN/GT-RAIN_val \
    $TRAIN_FLAGS $CLASSIFIER_FLAGS

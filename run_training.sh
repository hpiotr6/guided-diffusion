#!/bin/bash
TRAIN_FLAGS="--iterations 300
--anneal_lr True
--batch_size 64
--lr 3e-4
--save_interval 100
--weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256
--classifier_attention_resolutions 32,16,8
--classifier_depth 2
--classifier_width 64
--classifier_pool attention
--classifier_resblock_updown True
--classifier_use_scale_shift_norm True"

python3.10 \
    -m scripts.classifier_train \
    --data_dir datasets/GT-RAIN/GT-RAIN_train \
    --val_data_dir datasets/GT-RAIN/GT-RAIN_val \
    $TRAIN_FLAGS $CLASSIFIER_FLAGS

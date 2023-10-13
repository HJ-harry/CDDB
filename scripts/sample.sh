#!/bin/bash
python sample.py \
--ckpt sr4x-bicubic \
--n-gpu-per-node 1 \
--dataset-dir /media/harry/tomo/ImageNet/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC \
--batch-size 1 \
--use-fp16 \
--clip-denoise \
--nfe 100 \
--step-size 1.0
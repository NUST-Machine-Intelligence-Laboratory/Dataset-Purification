#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONWARNINGS="ignore"

export NET='resnet50_sub'
export path='model'
export data='/home/zcy/data/fg-web-data/web-bird'
export smooth=0

python train.py --net ${NET}  --path ${path} --data_base ${data} --step 1 --smooth ${smooth}


#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONWARNINGS="ignore"

export NET='resnet50_sub'
export path='model/step1'
export data='/home/zcy/data/fg-web-data/web-bird'

python train.py --net ${NET}  --path ${path} --data_base ${data} --step 1

sleep 100
python train.py --net ${NET}  --path ${path} --data_base ${data} --step 2

sleep 100
export path='model/step3'
export dl='dataset/clean_list_thr61.pkl'
python train.py --net ${NET}  --path ${path} --data_base ${data} --dl ${dl} --step 3

sleep 100
export NET='resnet18_ss'
export path='model/step3'
export dl='dataset/clean_list_thr61.pkl'
python train.py --net ${NET}  --path ${path} --data_base ${data} --dl ${dl} --step 3 --epoch 10

sleep 100
export NET='resnet50_sub'
export path='model/step4'
export dl='dataset/clean_list_thr61.pkl'
python train.py --net ${NET}  --path ${path} --data_base ${data} --dl ${dl} 'dataset/relabel_list_thr60_500.pkl' --step 4  --smooth 0.5
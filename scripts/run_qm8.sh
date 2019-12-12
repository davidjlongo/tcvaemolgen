#!/bin/bash

python -W ignore -m tcvaemolgen.train.train_prop --data data/01_raw/pa-graph/qm8 --hidden_size 160 --lr 1e-3 --distributed-backend dp --epochs 300 --depth 5 --gpus 0 --batch_size 64 --dropout 0.1 --n_heads 1 --multi True --n_classes 12 --n_rounds 10 --d_k 160 --dataset qm8

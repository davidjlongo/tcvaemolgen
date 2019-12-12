#!/bin/bash

python -W ignore -m tcvaemolgen.train.train_prop --data data/01_raw/pa-graph/qm7 --hidden_size 160 --lr 1e-3 --distributed-backend dp --n_workers 0 --epochs 100 --depth 5 --gpus 0 --batch_size 64 --dropout 0.2 --n_heads 1 --multi False --n_classes 12 --n_rounds 10 --d_k 160 --dataset qm7


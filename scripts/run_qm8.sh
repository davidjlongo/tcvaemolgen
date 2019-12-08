#!/bin/bash

python -W ignore -m tcvaemolgen.train.train_prop --data data/01_raw/pa-graph/qm8 --hidden_size 160 --lr 1e-3 --distributed-backend dp --epochs 100 --depth 5 --gpus 0 --batch_size 5000 --dropout 0.2 --n_heads 1 --multi True --n_classes 12 --n_rounds 10 --d_k 160 --dataset qm8

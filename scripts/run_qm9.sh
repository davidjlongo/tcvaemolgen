#!/bin/bash

python -W ignore -m tcvaemolgen.train.train_prop --data data/01_raw/pa-graph/qm9 --hidden_size 250 --lr 5e-4 --distributed-backend dp --epochs 200 --depth 5 --gpus 0 --batch_size 64 --dropout 0.2 --n_heads 2 --multi True --n_classes 12 --n_rounds 10 --d_k 250 --dataset qm9

#!/bin/bash

python -m tcvaemolgen.train.train_prop --data data/01_raw/pa-graph/bbbp --hidden_size 160 --lr 5e-4 --distributed-backend dp --epochs 150 --depth 5 --gpus 0 --batch_size 16 --dropout 0.2 --n_heads 2 --multi False --n_classes 1 --n_rounds 10 --d_k 80 --dataset bbbp --no_share True --batch_splits 2 --loss_type ce

#!/bin/bash

set -e

for batch_size in 100
do
    for lr in 0.0001
    do
        mkdir -p curves/cnn_$1
        mkdir -p reconstructed/cnn_$1
        mkdir -p generated/cnn_$1

        python3 vae_cnn.py --lr $lr --batch_size $batch_size

        mv *LC_cnn\.png curves/cnn_$1
        mv *\_x_cnn\.png reconstructed/cnn_$1
        mv *\_gen_cnn\.png generated/cnn_$1

    done
done

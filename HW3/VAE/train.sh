#!/bin/bash

set -e

for batch_size in 100
do
    for lr in 0.0001
    do
        mkdir -p curves/$1
        mkdir -p reconstructed/$1
        mkdir -p ground_truth/$1
        mkdir -p generated/$1

        python3 vae.py --lr $lr --batch_size $batch_size

        mv *LC\.png curves/$1
        mv *\_recon_x\.png reconstructed/$1
        mv *\_recon_x\.png reconstructed/$1
        mv *\_gen\.png generated/$1

    done
done

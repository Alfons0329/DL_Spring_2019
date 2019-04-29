#!/bin/bash
set -e
read -p "1: All combination of batch, 2: Current best " sel

if [ $sel -eq 1 ];
then
    for batch_size in 100
    do
        for learning_rate in 0.001
        do
            echo $learning_rate
            python3 cnn.py $learning_rate $batch_size 1 --vgg_small ada
        done
    done
else
    echo "do it later"
    #python3 dnn_1.py $1_16_0.00001 16 0.00001
fi

mkdir -p $1 #_P3
mv $1*\.png $1/ #_P3/

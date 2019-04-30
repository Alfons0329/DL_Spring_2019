#!/bin/bash
set -e
read -p "1: All combination of batch, 2: Current best " sel

if [ $sel -eq 1 ];
then
    for batch_size in 100
    do
        rm -rf my_vgg.pt best_acc.txt
        for learning_rate in 0.01 0.1
        do
            echo $learning_rate
            python3 cnn.py $learning_rate $batch_size 1 --vgg_small ada
        done
    done
else
    rm -rf my_vgg.pt best_acc.txt
    python3 cnn.py 0.01 200 1 --vgg_small ada
    rm -rf my_vgg.pt best_acc.txt
    python3 cnn.py 0.1 100 1 --vgg_small ada
fi

mkdir -p $1 #_P3
mv *.png $1/ #_P3/

#!/bin/bash
set -e
read -p "1: All combination of batch, 2: Current best 3: Run full VGG16 if you have GPU better than RTX2080Ti " sel

if [ $sel -eq 1 ];
then
    for batch_size in 100
    do
        rm -rf my_vgg.pt best_acc.txt
        for learning_rate in 0.01 0.1
        do
            echo $learning_rate
            python3 cnn.py $learning_rate $batch_size 1 --vgg_small $2
        done
    done
elif [ $sel -eq 2 ];
then
    rm -rf my_vgg.pt best_acc.txt
    #python3 cnn.py 0.01 200 1 --vgg_small $2
    rm -rf my_vgg.pt best_acc.txt
    #python3 cnn.py 0.1 100 1 --vgg_small $2
else
    python3 cnn_vgg16.py 0.01 40 1 --vgg_normal $2
fi

mkdir -p $1\_$2 #_P3
mv *.png $1\_$2/ #_P3/

#!/bin/bash
set -e
read -p "1: Compare stride size, 2: Compare kernel size, 3: Current best 4: Run full VGG16 if you have GPU better than RTX2080Ti " sel

if [ $sel -eq 1 ];
then
    echo "Compare the effect of strid_size"
    for batch_size in 16
    do
        for strid_size in 1 3
        do
            rm -rf my_vgg.pt best_acc.txt
            python3 cnn.py 0.01 $batch_size $strid_size 2 --vgg_small $2
        done
    done
elif [ $sel -eq 2 ];
then
    echo "Compare the effect of kernel_size"
    for batch_size in 16
    do
        for kernel_size in 3 5
        do
            rm -rf my_vgg.pt best_acc.txt
            python3 cnn.py 0.01 $batch_size 2 $kernel_size --vgg_small $2
        done
    done

elif [ $sel -eq 2 ];
then
    rm -rf my_vgg.pt best_acc.txt
    python3 cnn.py 0.01 64 2 2 --vgg_small $2
    #python3 cnn.py 0.01 200 1 --vgg_small $2
    #python3 cnn.py 0.1 64 1 --vgg_small $2
else
    python3 cnn_vgg16.py 0.01 32 1 --vgg_normal sgd
    python3 cnn.py 0.01 32 2 2 --vgg_small sgd
fi

mkdir -p $1\_$2
mv *.png $1\_$2/
mv *.pt best_acc.txt $1\_$2/

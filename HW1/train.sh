#!/bin/bash
read -p "1: All combination of batch, 2: Current best " sel

if [ $sel -eq 1 ];
then
    for batch_size in 4 8 16 20 25 32 40
    do
        for learning_rate in  0.0000001 0.0000003 0.0000005 0.000001 0.000003 0.000005
        do
            echo $learning_rate
            python3 dnn_1.py $1\_$batch_size\_$learning_rate $batch_size $learning_rate
        done
    done
else
    python3 dnn_1.py $1_16_0.00001 16 0.00001
fi

mkdir -p $1 #_P3
mv $1*\.png $1/ #_P3/

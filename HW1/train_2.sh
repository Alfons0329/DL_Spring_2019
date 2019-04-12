#!/bin/bash
read -p "1: All combination of batch, 2: Current best " sel

if [ $sel -eq 1 ];
then
    for batch_size in 16 20 25 32 40
    do
        for learning_rate in  0.0000001 0.0000003 0.0000005 0.000001 0.000003 0.000005
        do
            echo $learning_rate
            python3 dnn_2.py $1\_$batch_size\_$learning_rate $batch_size $learning_rate
        done
    done
else
    #python3 dnn_2.py $1_16_0.1 32 0.1
    #python3 dnn_2.py $1_16_0.01 32 0.01
    python3 dnn_2.py $1_16_0.001 32 0.001
    python3 dnn_2.py $1_16_0.0001 32 0.0001
fi

mkdir -p $1 #_P3
mv $1*\.png $1/ #_P3/

#!/bin/bash
read -p "1: All combination of batch, 2: Current best " sel
if [ $sel -eq 1 ];
then
    for batch_size in 4 8 16 20 25 32 40 50
    do
        for learning_rate in 0.000001 0.00001 0.0001 0.001 0.01 0.1 0.5
        do
            echo $batch_size $learning_rate
            python3 dnn_1.py $1\_$batch_size\_$learning_rate $batch_size $learning_rate
        done
    done
else
    python3 dnn_1.py $1_32_0.00001 32 0.00001
fi

#!/bin/bash
for batch_size in 4 8 16 32 40 50
do
    for learning_rate in 0.000001 0.00001 0.0001 0.001 0.01 0.1 0.5 1 2 4
    do
        echo $batch_size $learning_rate
        python3 dnn_1.py $1\_$batch_size\_$learning_rate $batch_size $learning_rate
    done
done

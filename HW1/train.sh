#!/bin/bash
read -p "1: All combination of batch, 2: Current best " sel

if [ $sel -eq 1 ];
then
    batch_size=$2
    for learning_rate in 0.0001 0.001 0.01 0.1 1
    do
        echo $learning_rate
        python3 dnn_1.py $1\_$batch_size\_$learning_rate $batch_size $learning_rate
    done
else
    python3 dnn_1.py $1_8_0.000005 8 0.000005
fi

mkdir -p $1_1
mv $1*\.png $1_1/

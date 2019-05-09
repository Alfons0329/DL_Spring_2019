#!/bin/bash
read -p "1: All combination of batch, 2: Current best " sel

if [ $sel -eq 1 ];
then
    for batch_size in 16 128  # small big
    do
        for learning_rate in 0.00001 0.0001 0.001
        do
            for method in adam sgd
            do
                python3 rnn.py $learning_rate $batch_size $method
                mkdir -p $method\_$learning_rate\_$batch_size
                mv $method*\.png $method\_$learning_rate\_$batch_size
            done
        done
    done
else
    python3 dnn_1.py $1_16_0.00001 16 0.00001
fi


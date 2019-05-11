#!/bin/bash
read -p "1: RNN all combinations, 2: RNN current best, 3: LSTM all combinations " sel

if [ $sel -eq 1 ];
then
    mkdir -p RNN_result/$1
    for batch_size in 8 256  # small big
    do
        for learning_rate in 0.00001 0.0001 0.001 0.01
        do
            for method in adam sgd
            do
                python3 rnn.py $learning_rate $batch_size $method
                mv $method*\.png RNN_result/\_$1
            done
        done
    done
elif [ $sel -eq 2 ];
then
    python3 rnn.py 0.0001 128 sgd
elif [ $sel -eq 3 ];
then
    mkdir -p LSTM_result
    mkdir -p LSTM_result/$1
    for batch_size in 16 128  # small big
    do
        for learning_rate in 0.00001 0.0001 0.001
        do
            for method in adam sgd
            do
                python3 lstm.py $learning_rate $batch_size $method
                mv $method*\.png LSTM_result/\_$1
            done
        done
    done
fi


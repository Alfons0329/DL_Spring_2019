#!/bin/bash
set -ex
echo $#
if [ $# -ne 1 ];
then
    echo "Usage: ./train.sh {date}"
    exit
fi

read -p "1: Train, 2: Test " what

mkdir -p graph
mkdir -p graph/$1
rm -rf output/*
if [ $what -eq 1 ];
then
	for lr in 0.0002
	do
		python3 cyclegan_train.py --cuda --lr $lr
        mv *\_dis\.png graph/$1
        mv *\_gen\.png graph/$1
    done
else
	for lr in 0.0002
	do
		python3 cyclegan_test.py --cuda --lr $lr
	done
fi

#!/bin/bash
set -ex
read -p "1: Train, 2: Test " what

if [ $what -eq 1 ];
then
	for lr in 0.0002
	do
		python3 cyclegan_train.py --cuda --lr $lr 
	done
else
	for lr in 0.0002
	do
		python3 cyclegan_test.py --cuda --lr $lr 
	done
then
fi

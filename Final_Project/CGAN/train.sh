#!/bin/bash
set -ex
echo $#
if [ $# -ne 3 ];
then
    echo "Usage: ./train.sh {date} {epochs} {batch_size}"
    exit
fi

read -p "1: Train + test, 2: Test only" what

mkdir -p graph
mkdir -p graph/$1
rm -rf output/*

if [ $what -eq 1 ];
then
	for lr in 0.0002
	do
        # set your own path for young and old
		python3 cyclegan_train.py --cuda --lr $lr --epochs $2 --batch_size $3 --old_root ../alfons/DL_Final_Train_Data/old_face --young_root ../alfons/DL_Final_Train_Data/young_face
        mv *\_dis\.png graph/$1
        mv *\_gen\.png graph/$1
		#python3 cyclegan_test.py --cuda --lr $lr --batch_size 5 --output_imgs 20
		python3 cyclegan_test.py --cuda --lr $lr --batch_size 5 --output_imgs 20
    done
else
	for lr in 0.0002
	do
		python3 cyclegan_test.py --cuda --lr $lr --batch_size 10 --output_imgs 10
	done
fi

#!/bin/bash

read -p "1: Send HW to friend, 2: Send to NCTU E3 " what

if [ $what -eq 1 ];
then
    tar -cvf CGAN.tar animation cartoon *.py train.sh
else
    zip HW3_0416324.zip *.py *.pdf
fi

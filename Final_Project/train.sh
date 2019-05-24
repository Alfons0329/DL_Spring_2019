#!/bin/bash

if [ $# -ne 2 ];
then
    echo "Usage; ./train.sh <path/to/content_img> <id of content_img>"
    echo "Example: ./train.sh content_img/c_9_face.png 9"
    exit
fi

read -p "Style with main face part only? 1 y 2 n: " face
if [ $face -eq 1 ];
then
    style_cnt=1
    # todo_pattern="style_img/.*\_face\.png"
    # for f in $todo_pattern

    for f in style_img/*\_face\.png;
    do
        echo "Using style: " $f
        if [ ! -e $f ];
        then
            echo "File not exist! "
        fi
        for steps in 25 100;
        do
            python3 main.py --style_img $f --content_img $1 --steps $steps --style_cnt $style_cnt --content_cnt $2
        done
        style_cnt=$(($style_cnt+1))
    done
else

    # todo_pattern="style_img/.*\.png"
    # for f in $todo_pattern;

    for f in style_img/*\.png
    do
        # dont_pattern="style_img/.*\_face\.png"
        if [ $f != style_img/*\_face\.png ];
        then
            echo "Using style: " $f
            python3 main.py --style_img $f
        fi
    done
fi

mv *.png output_img/

#!/bin/bash

read -p "Style with main face part only? 1 y 2 n: " face
if [ $face -eq 1 ];
then
    style_cnt=1
    steps=75
    content_cnt=7
    for f in style_img/*\_face.png;
    do
        echo "Using style: " $f
        python3 main.py --style_img $f --content_img $1 --steps $steps --style_cnt $style_cnt --content_cnt $content_cnt
        style_cnt=$(($style_cnt+1))
    done
else
    for f in style_img/*.png;
    do
        pattern="*\_face.png"
        if [ $f -ne $pattern];
        then
            echo "Using style: " $f
            python3 main.py --style_img $f
        fi
    done
fi

mv *.png output_img/

#!/bin/bash
set -e
#if [ $# -ne 2 ];
#then
#    echo "Usage; ./train.sh path/to/content_img id_of_content_img"
#    echo "Example: ./train.sh content_img/c_9_face.png 9"
#    exit
#fi

read -p "Style with main face part only? 1 y 2 n: " face
if [ $face -eq 1 ];
then
    # todo_pattern="style_img/.*\_face\.png"
    # for f in $todo_pattern

    for content in content_img/*g;
    do
        if [ ! -e $content ];
        then
            echo "File not exist! "
        fi
        for style in style_img/*\_face\.png;
        do
            if [ ! -e $style ];
            then
                echo "File not exist! "
            fi
            content_id=$(echo $content | sed 's/content_img\/c_//g')
            content_id=$(echo $content_id | sed 's/.jpg//g; s/.png//g; s/.jpeg//g')

            style_id=$(echo $style | sed 's/style_img\/s_//g')
            style_id=$(echo $style_id | sed 's/\_face//g')
            style_id=$(echo $style_id | sed 's/.jpg//g; s/.png//g; s/.jpeg//g')

            echo "Content: " $content " with id " $content_id " Style: " $style " with id " $style_id
            if [[ $content != *$content_id* ]];
            then
                echo "ID and content image mismatch, plese fix this shell script"
                exit
            fi
            python3 main.py --style_img $style --content_img $content --steps 25 --style_cnt $style_id --content_cnt $content_id
            style_id=$(($style_id+1))
        done
        mkdir -p output_img/c$content_id
        mv s*c$content_id*\.png output_img/c$content_id
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


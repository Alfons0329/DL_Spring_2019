#!/bin/bash
if [ $# -ne 1 ];
then
    echo "Usage: ./style_face.sh <parh/to/dir/for/face/crop>"
    exit
fi

cd $1

style_cnt=$(ls | wc -l)
style_cnt=$((style_cnt-2))
index=1
# todo_pattern=*{jpg,jpeg}
# dont_pattern=*\_face\.{.png}

for f in *{jpg,jpeg};
do
    if [ $f != *\_face\.{.png} ];
    then
        echo "f: " $f
	    python3 face_detection.py $f
    fi
done

cd ..

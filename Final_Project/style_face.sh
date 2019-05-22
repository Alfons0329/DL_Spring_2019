#!/bin/bash
cd style_img

style_cnt=$(ls | wc -l)
index=1

for f in *.jpg;
do
	mv "$f" s\_$index.jpg
	index=$((index+1))
done

for f in *.jpg;
do
	python3 ../face_detection.py $f
done

cd ..

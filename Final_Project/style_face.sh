#!/bin/bash
cd style_img

style_cnt=$(ls | wc -l)
style_cnt=$((style_cnt-3))
index=1

for f in $(seq 1 $style_cnt);
do
	python3 face_detection.py s\_$f
done

cd ..

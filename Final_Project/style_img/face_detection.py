import cv2
import os, sys
p = sys.argv[1]
def face_detect(path):
	img = cv2.imread(path)
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)
	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.imwrite(sys.argv[1] + '_face.png', img[y:y+h,x:x+w])

print(p + '.jpg')
face_detect(p + '.jpg')

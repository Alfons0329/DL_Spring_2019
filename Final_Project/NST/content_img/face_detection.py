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
		f_name = os.path.splitext(sys.argv[1])[0] + '_face.png'
		print('change ', sys.argv[1], ' to ', f_name)
		cv2.imwrite(f_name, img[y:y+h,x:x+w])

print(p)
face_detect(p)

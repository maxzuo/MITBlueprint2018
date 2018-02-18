import sys
import numpy
sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import os
import datetime

#try threading this
def speak(word):
	os.system("say \'" + str(word) + "\'")

fistCascade = cv2.CascadeCalssifier("fist.xml")

cap = cv2.VideoCapture(0)


#Check for fists and everything to do with OpenCV
while True:

	_, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if type(fists) == type(None):
		fists = []

	fist = []

	for (x, y, width, height) in fists:
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 240, 0), 2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
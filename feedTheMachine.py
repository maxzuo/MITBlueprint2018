import sys
import numpy as np
sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import os
import datetime
from scipy.stats import itemfreq

#try threading this

def dominant(fist):
	global lower, upper
	arr = np.float32(fist)
	# print arr.shape
	# print arr
	# print fist
	pixels = arr.reshape((-1, 3))

	k = 2

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	_, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, flags)

	palette = np.uint8(centroids)
	quantized = palette[labels.flatten()]
	quantized = quantized.reshape(fist.shape)

	# dominantColor = palette[np.argmax(itemfreq(labels)[:, -1])]

	if sum(palette[0].flatten()) < sum(palette[1].flatten()):
		lower = np.array([palette[0][0] * 0.8, palette[0][1] * 0.8, palette[0][2]*0.8], dtype = "uint8")
		upper = np.array([palette[1][0] * 1.2, palette[1][1] * 1.2, palette[1][2] * 1.2], dtype = "uint8")
	else:
		lower = np.array([palette[1][0] * 0.8, palette[1][1] * 0.8, palette[1][2] * 0.8], dtype = "uint8")
		upper = np.array([palette[0][0] * 1.2, palette[0][1] * 1.2, palette[0][2] * 1.2], dtype = "uint8")


def speak(word):
	os.system("say \'" + str(word) + "\'")

fistCascade = cv2.CascadeClassifier("fist.xml")

lower = None
upper = None

cap = cv2.VideoCapture(0)


#Check for fists and everything to do with OpenCV
while True:

	_, frame = cap.read()

	width, height, _ = frame.shape

	frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
	# color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	fists = fistCascade.detectMultiScale(gray, 1.3, 5)
	
	if type(fists) != type(None):
		# print len(fists)

		fist = []
		size = None

		for (x, y, width, height) in fists:
			# cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 240, 0), 2)
			# ROI is [small y: big y, smallx: bigX]

			fist = frame[y: y+height, x: x + width]
			size = (x, y, width, height)

			#Find most dominant of colors
			dominant(fist)
			# print itemfreq(labels)

			# print dominantColor

			#use dominantColor to threshold
			

			

			break
	if type(lower) != type(None):
		threshold = cv2.inRange(frame, lower, upper)
		cv2.imshow("threshold", threshold)
	cv2.imshow("frame", frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
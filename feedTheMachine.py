import sys
import numpy as np
sys.path.append("/usr/local/lib/python2.7/site-packages")
import cv2
import os
import datetime
from scipy.stats import itemfreq

#try threading this

# params = cv2.SimpleBlobDetector_Params()

# params.filterByInertia = False
# params.filterByConvexity = False
# params.filterByCircularity = False

# params.filterByColor = True

# detector = cv2.SimpleBlobDetector_create(params)
center = (0, 0)

def getClosestContour(frame):
	global center
	_, contours, _ = cv2.findContours(frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	distance = 100000

	# handSize = 0
	handCenter = (0, 0)
	handInfo = None
	if (len(contours) != 0):
		handContour = contours[0]
		for contour in contours:

			x, y, width, height = cv2.boundingRect(contour)

			currentDistance = (((center[0] - (x + width/2)) ** 2 + (center[1] - (y + height /2 ))) ** 2)
			if currentDistance < distance:
				distance = currentDistance
				handContour = contour
				handCenter = ((x + width/2), (y + height/2))
				handInfo = [x, y, width, height]
		center = handCenter
		if handInfo is not None:
			return handContour, handInfo
		else:
			return None
	else:
		return None


	# 		size = cv2.contourArea
	# 		if type(contour) != int:
	# 		centerX, centerY = getContourMoment(contour)
	# 		fistCenterX, fistCenterY = center[0], center[1]
	# 		if ((centerX - fistCenterX) ** 2 + (centerX - fistCenterX) ** 2):
	# 			handContour = contour
	# 			handCenter = (fistCenterX, fistCenterY)
	# 			handSize = size
	# return handCenter, handSize


# def getContourMoment(contour):
# 	print type(contour)
# 	print contour
# 	m = cv2.moments(contour)
# 	centerX = int(m['m10']/m['m00'])
# 	centerY = int(m['m01']/m['m00'])
	
# 	return (centerX, centerY)

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
		lower = np.array([palette[0][0] * 0.9, palette[0][1] * 0.9, palette[0][2] * 0.9], dtype = "uint8")
		upper = np.array([palette[1][0] * 1.1, palette[1][1] * 1.1, palette[1][2] * 1.1], dtype = "uint8")
	else:
		lower = np.array([palette[1][0] * 0.9, palette[1][1] * 0.9, palette[1][2] * 0.9], dtype = "uint8")
		upper = np.array([palette[0][0] * 1.1, palette[0][1] * 1.1, palette[0][2] * 1.1], dtype = "uint8")

def findCenter(x, y, width, height):
	return (x + width/2, y + height/2)


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

			fistCenter = findCenter(x, y, width, height)
			center = fistCenter
			

			break

	#use dominant colors to threshold
	if type(lower) != type(None):
		threshold = cv2.inRange(frame, lower, upper)

		# keyPoints = detector.detect(cv2.bitwise_not(threshold))
		# keyPointsDrawn = cv2.drawKeypoints(frame, keyPoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		# radius = (handSize ** 0.5) * 1.3
		# final = frame.copy()
		# cv2.rectangle(final, (handCenter[0] - radius, handCenter[1] - radius), (handCenter[0] + radius, handCenter[1] + radius), (0, 0, 255), 4)


		kernel = np.ones((4, 4), dtype = np.uint8)
		filtered = cv2.erode(threshold, kernel, iterations=1)

		kernel = np.ones((9, 9), dtype = np.uint8)
		filtered = cv2.dilate(filtered, kernel, iterations=1)
		
		final = frame.copy()
		
		result = getClosestContour(filtered)
		if type(result) != type(None):
			handContour, handInfo = result
			x, y, w, h = handInfo
			cv2.rectangle(final, (x, y), (x + w, y + h), (255, 0, 255), 5)


		cv2.imshow("threshold", filtered)
		cv2.imshow("final", final)

	cv2.imshow("frame", frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
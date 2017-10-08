import cv2
import numpy as np
import math
import operator


def check_cataract(img_str):
	img = cv2.imread(img_str)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	grayscale = gray
	ret, gray = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
	gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
	gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
	threshold = cv2.inRange(gray, 250, 255)
	contours, heirarchy = cv2.findContours(
		threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
	)
	c = None
	max = 0
	for contour in contours:
		if cv2.contourArea(contour) > max:
			max = cv2.contourArea(contour)
			c = contour
	center = cv2.moments(c)
	r = math.ceil((cv2.contourArea(c) / 3.14) ** 0.5)
	img2 = np.zeros_like(grayscale)
	cx = int(center['m10'] / center['m00'])
	cy = int(center['m01'] / center['m00'])
	cv2.circle(img2, (cx, cy), int(r), (255, 255, 255), -1)
	res = cv2.bitwise_and(grayscale, img2)
	resized = cv2.resize(res, (256, 256))
	mean, std = cv2.meanStdDev(resized)
	mean = mean[0][0]
	std = std[0][0]

	# U = abs((1 - std / mean))
	# if U > 1: U //= 10
	H = cv2.calcHist( [gray.astype('float32')],
              channels=[0],
              mask=np.ones_like(gray).astype('uint8'),
              histSize=[256],
              ranges=[0, 256])
	U = 0
	pixels = gray.shape[0] * gray.shape[1]
	for h in H:
		U += ((h[0] / pixels) ** 2)

	content = {"healthy": 0, "severe": 0, "mild": 0}

	if U < 0.5000: content["healthy"] += 1
	elif U > 0.5001: content["severe"] += 2
	else: content["mild"] += 1

	if mean > 134: content["severe"] += 1
	elif mean < 125: content["healthy"] += 1
	else: content["mild"] += 1

	if std > 3.18: content["severe"] += 1
	elif std < 1.59: content["healthy"] += 1
	else: content["mild"] += 1

	return content
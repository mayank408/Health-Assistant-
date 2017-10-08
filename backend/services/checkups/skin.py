import cv2
import numpy as np
import base64


def check_cancer(img_str):
	"""
	data = base64.decodeString(img_str)
	if ".png" in data:
		extension = ".png"
	else:
		extension = ".jpg"
	with open("test" + extension, "wb") as f:
		f.write(data)
	img = cv2.imread("test" + extension)
	"""
	print img_str
	img = cv2.imread(img_str)
	img_copy = img
	R = np.mean(img[:, :, 0]) / 1000
	G = np.mean(img[:, :, 1]) / 1000
	B = np.mean(img[:, :, 2]) / 1000

	const = [
		[0.412453, 0.357580, 0.180423],
		[0.212671, 0.715160, 0.072169],
		[0.019334, 0.119193, 0.950227]
	]

	X, Y, Z = np.dot(const, [[R], [G], [B]])

	L = 116 * Y ** (1.0 / 3) - 16
	if Y <= 0.008856:
		L = 903.3 * Y

	H = 60 * (B - G) / (max(R, G, B) - min(R, G, B))

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(
		img, min(H, L), max(H, L), cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	)
	cv2.bitwise_and(img, img, mask=mask)

	contours, heirarchy = cv2.findContours(
		mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
	)

	contour = max(contours, key=cv2.contourArea)
	area = cv2.contourArea(contour) / (150.0 * 150.0)
	# Diameter
	D = ((area / 3.14) ** 0.5) * 2
	if D > 6:
		D = 5

	ellipse = cv2.fitEllipse(contour)
	a = ellipse[1][0]
	b = ellipse[1][1]
	# Asymmetrix Index
	A = (3.14 * a * b - (3.14 * a * a / 4) - (3.14 * b * b / 4))
	A = A / (2 * cv2.contourArea(contour))

	# Border Irregularity
	B = cv2.arcLength(contour, 1) / 300

	rr, gg, bb = np.uint8(
		np.average(
			np.average(cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV), axis=0), axis=0
		)
	)
	# Color Index
	C = (rr + gg + bb) / 9.0

	print A, B, C, D

	TDS = 1.3 * A + 0.1 * B + 0.5 * C + 0.5 * D
	print TDS
	if TDS < 6.25 and C > 5:
		return "benign"
	elif TDS > 7.25 or C < 5:
		return "malignant melanoma"
	else:
		return "suspicious lesion"

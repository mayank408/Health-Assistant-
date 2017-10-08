import cv2


class Capture():
	def __init__(self, source):
		error = 1
		while error == 1:
			try:
				self.capture = cv2.VideoCapture(source)
				error = 0
			except:
				pass

	def capture_frame(self):
		frame = self.capture.read()[1]
		while frame is None:
			print "Error in retreiving frame..."
			print "Trying again..."
			frame = self.capture.read()[1]
		return frame

	def shutdown(self):
		self.capture.release()
		cv2.destroyAllWindows()

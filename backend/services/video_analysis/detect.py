from capture import Capture
from processing import Analyzer
import cv2

source = 0
xx = 1
flag = True


class Detect():
	def __init__(self):
		self.capture = Capture(source)
		self.analyzer = Analyzer()

	def main(self):
		global flag
		global xx
		if self.analyzer.face and xx:
			self.analyzer.face = False
			xx = 0
			self.analyzer.find = not self.analyzer.find
		frame = self.capture.capture_frame()
		self.analyzer.incoming_frame = frame
		self.analyzer.analyze()
		out_frame = self.analyzer.outgoing_frame
		cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty(
			"window",
			cv2.WND_PROP_FULLSCREEN,
			cv2.cv.CV_WINDOW_FULLSCREEN
		)
		cv2.imshow("window", out_frame)
		c = cv2.waitKey(10) & 255
		if c == 27:
			self.capture.shutdown()
		if self.analyzer.cnt == 31:
			flag = False
			self.capture.shutdown()
			# print self.analyzer.avg_bpm
			# print self.analyzer.breathes
			# return beats per minute and breath rate per minute
			return self.analyzer.avg_bpm, self.analyzer.breathes

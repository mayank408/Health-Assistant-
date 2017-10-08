import numpy as np
import time
import cv2
from scipy.signal import argrelextrema


class Analyzer():
	def __init__(self):
		self.avg_bpm = self.cnt = self.fps = self.bpm = self.breathes = 0
		self.up = self.down = -1
		self.index = 1
		self.incoming_frame = self.outgoing_frame = np.zeros((10, 10))
		self.face = False
		self.find = True
		self.buffer_size = 250
		self.buffer_data = []
		self.times = []
		self.t_times = []
		self.freqs = []
		self.fft = []
		self.face_rect = []
		self.previous_frame = None
		self.cascade = cv2.CascadeClassifier("frontface_cascade.xml")
		self.last_fft_center = np.array([0, 0])
		self.hsv = None
		self.t0 = time.time()

	# shift zero frequency to center
	def fft_shift(self, detected_forehead):
		x, y, w, h = detected_forehead
		center = np.array([x + 0.5 * w, y + 0.5 * h])  # approximation
		shift = np.linalg.norm(center - self.last_fft_center)
		self.last_fft_center = center
		return shift

	# get absolute coordinates of forehead
	def get_forehead_coords(self, head_rel_x, head_rel_y, head_w, head_h):
		try:
			x, y, w, h = self.face_rect
		except:
			# Boo :P
			x = y = w = h = 0
		# making PEP8 compilant :D
		return [int(x + w * head_rel_x - (w * head_w / 2.0)),
										int(y + h * head_rel_y - (h * head_h / 2.0)),
										int(w * head_w), int(h * head_h)]

	# I hope name is enough to explain what function does :P (I assume)
	def get_mean_intensity(self, boundary_coords):
		x, y, w, h = boundary_coords
		# stripping pixel frame of forehead
		forehead_frame = self.incoming_frame[y:y + h, x:x + w, :]
		I0 = np.mean(forehead_frame[:, :, 0])
		I1 = np.mean(forehead_frame[:, :, 1])
		I2 = np.mean(forehead_frame[:, :, 2])
		return (I0 + I1 + I2) / 3

	# all the main calculations will be done here
	def analyze(self):
		try:
			self.times.append(time.time() - self.t0)
			self.outgoing_frame = self.incoming_frame
			# Well, doesn't everyone like grayscale? :D
			self.grayscale = cv2.cvtColor(self.incoming_frame, cv2.COLOR_BGR2GRAY)
			# contrasting the image
			self.grayscale = cv2.equalizeHist(self.grayscale)
			if self.find:
				self.buffer_data = []
				self.times = []
				detected_faces = list(self.cascade.detectMultiScale(
					self.grayscale,
					scaleFactor=1.3,
					minNeighbors=4,
					flags=cv2.CASCADE_SCALE_IMAGE,
					minSize=(50, 50)
				))
				if len(detected_faces):
					# sorted accoding to max area of detected face
					detected_faces.sort(key=lambda z: z[-1] * z[-2])
					if self.fft_shift(detected_faces[-1] > 10):
						self.face_rect = detected_faces[-1]
						self.face = True
				forehead = self.get_forehead_coords(0.5, 0.18, 0.25, 0.15)
				try:
					x, y, w, h = self.face_rect
				except:
					x = y = w = h = 0
				cv2.rectangle(self.outgoing_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
				try:
					x, y, w, h = forehead
				except:
					x = y = w = h = 0
				cv2.rectangle(self.outgoing_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
				return

			forehead = self.get_forehead_coords(0.5, 0.18, 0.25, 0.15)
			try:
				x, y, w, h = forehead
			except:
				x = y = w = h = 0
			cv2.rectangle(self.outgoing_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

			mean_value = self.get_mean_intensity(forehead)
			self.buffer_data.append(mean_value)
			length = len(self.buffer_data)
			data = np.array(self.buffer_data)
			# stabilizing ...
			if length > 10:
				self.fps = float(length) / (self.times[-1] - self.times[0])
				even_intervals = np.linspace(self.times[0], self.times[-1], length)
				# hamming is good for DTFT
				interpolation = np.hamming(length) * np.interp(
					even_intervals, self.times, data
				)
				# ^, PEP8 compilant :D
				interpolation -= np.mean(interpolation)
				variation = np.fft.rfft(interpolation)
				phase = np.angle(variation)
				self.fft = np.abs(variation)
				self.freqs = float(self.fps) / length * np.arange(length / 2 + 1)
				freqs = 60. * self.freqs
				index = np.where((freqs > 50) & (freqs < 240))

				pruned = self.fft[index]
				phase = phase[index]

				self.freqs = freqs[index]
				self.fft = pruned

				temp_index = np.argmax(pruned)
				t = (np.sin(phase[temp_index]) + 1.0) / 2
				t = 0.9 * t + 0.1
				theta1, theta2 = t, 1 - t

				self.bpm = self.freqs[temp_index]
				self.index += 1

				R = theta1 * self.incoming_frame[y:y + h, x:x + w, 0]
				G = theta1 * self.incoming_frame[y:y + h, x:x + w, 1] + \
					theta2 * self.grayscale[y:y + h, x:x + w]
				B = theta1 * self.incoming_frame[y:y + h, x:x + w, 2]
				self.outgoing_frame[y:y + h, x:x + w] = cv2.merge([R, G, B])
				time_interval = (self.buffer_size - length) / self.fps
				display = "%0.1f bpm, %0.0f seconds" % (self.bpm, time_interval)
				if self.bpm > 50 and self.bpm < 110:
					# Breath Rate (detecting movement of chest)
					if self.cnt == 0:
						self.previous_frame = cv2.cvtColor(
							self.incoming_frame, cv2.COLOR_BGR2GRAY
						)
						self.hsv = np.zeros_like(self.incoming_frame)
						self.hsv[..., 1] = 255
					else:
						next_frame = cv2.cvtColor(self.incoming_frame, cv2.COLOR_BGR2GRAY)
						flow = cv2.calcOpticalFlowFarneback(
							self.previous_frame, next_frame, 0.5, 3, 15, 3, 5, 1.2, 0
						)
						mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
						self.hsv[..., 0] = ang * 180 / np.pi / 2
						self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

						maxima = argrelextrema(self.hsv, np.greater)
						minima = argrelextrema(self.hsv, np.less)

						mean1, mean2 = 0, 0
						for x in maxima[1]:
							mean1 += x
						mean1 = mean1 / len(maxima[1])
						for x in minima[1]:
							mean2 += x
						mean2 = mean2 / len(minima[1])

						if mean1 > mean2:
							if self.up == -1:
								self.up = 1
							elif self.down == 1:
								self.up = 1
								self.dowm = 0
								self.breathes += 1
						elif mean1 < mean2:
							if self.up == 1:
								self.up = 0
								self.down = 1
								self.breathes += 1

						self.previous_frame = next_frame

					self.cnt += 1
					# print self.cnt
					self.avg_bpm = (self.avg_bpm * (self.cnt - 1) + self.bpm) / self.cnt
				cv2.putText(
					self.outgoing_frame,
					display,
					(x - w / 2, y),
					cv2.FONT_HERSHEY_PLAIN,
					1,
					(0, 0, 255)
				)
		except:
			pass

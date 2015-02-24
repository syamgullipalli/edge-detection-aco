__author__ = 'syam'

import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

class ParseImage(object):
	"""Parse the image"""

	def __init__(self, **kwargs):
		self.file_path = kwargs['file_path']
		self.debug      = kwargs['debug']
		self.heu_coords = [[(-1, -1),(1,1)],
						   [(-1, 0), (1, 0) ],
						   [(-1, 1), (1, -1)],
						   [(0, -1), (0, 1)]]

		# Set logging
		log.propagate = self.debug

	def parseIntensity(self):
		log.debug("Parsing image ... START")

		# Read image as gray scale
		img = cv2.imread(self.file_path, 0)

		# Smooth using Gaussian filter
		# img = cv2.GaussianBlur(img, (3, 3), 0)

		# Show image
		# cv2.imshow('Original Image Smoothed by Gaussian Blur', img)
		# cv2.waitKey(500)


		self.intensity_matrix = np.array(img, dtype='int64')

		# Image size
		(self.x_max_index, self.y_max_index) = self.intensity_matrix.shape

		log.debug("Parsing image ... DONE")

		# Save to text
		np.savetxt('gray_values.txt', self.intensity_matrix, fmt='%d', newline='\n')
		log.debug("Image intensity matrix wrote to gray_values.txt")

		# Heuristic matrix initialization
		self.heuristic_matrix = np.zeros(shape=(self.x_max_index, self.y_max_index), dtype='int64')

		# Calculate heuristic matrix
		log.debug("Calculating heuristic matrix ... START")
		for (x_index, y_index), intensity in np.ndenumerate(self.intensity_matrix):
			self.__heuristic(x_index = x_index, y_index = y_index)
		log.debug("Calculating heuristic matrix ... DONE")

		# Save heuristic matrix to file
		np.savetxt('heuristic_matrix.txt', self.heuristic_matrix, fmt='%d', newline='\n')
		log.debug("Heuristic matrix wrote to heuristic_matrix.txt")
#		cv2.imshow('Image', intensity_matrix)
#		cv2.waitKey(0)

		return self.heuristic_matrix


	def __heuristic(self, x_index, y_index):
		V_c = 0
		for index, coord in enumerate(self.heu_coords):
			if(	(0 <= x_index + coord[0][0] < self.x_max_index) and
				(0 <= y_index + coord[0][1] < self.y_max_index) and
				(0 <= x_index + coord[1][0] < self.x_max_index) and
				(0 <= y_index + coord[1][1] < self.y_max_index) ):
				V_c = V_c + abs(self.intensity_matrix[x_index + coord[0][0]][y_index + coord[0][1]] - self.intensity_matrix[x_index + coord[1][0]][y_index + coord[1][1]])

		self.heuristic_matrix[x_index][y_index] = V_c

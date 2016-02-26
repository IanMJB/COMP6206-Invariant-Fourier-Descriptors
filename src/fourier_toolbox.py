# Future imports.
from __future__ import division

# External libraries.
import copy
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial import distance

class fourier_toolbox:

	# Reads in and returns the image in OpenCV format.
	def read_image(self, img_dir, img_name):
		return cv2.imread(img_dir + img_name, 0)

	# Converts an RGB image into greyscale.
	# Then converts the greyscale image into a binary one, and returns the threshold image.
	def apply_thresholding(self, image, is_img_rgb):
		img_clone	= self.get_copy(image)

		if is_img_rgb:
			img_clone	= cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)

		ret, threshold	= cv2.threshold(img_clone, 127, 255, cv2.THRESH_BINARY)

		return threshold

	# Finds and returns the contours.
	# Potential TODO: Expand to other parameters:
	# Most exhaustive parameters fed in, change later if possible - specifically CHAIN_APPROX_NONE to CHAIN_APPROX_SIMPLE.
	def find_contours(self, image):
		img_clone	= self.get_copy(image)

		# TODO: Measure speed gains and discuss in slides
		contours, hierarchy	= cv2.findContours(img_clone, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		return contours

	# Returns a clone of an image, or the original image if no argument passed.
	def get_copy(self, image):
		return copy.deepcopy(image)

	# Normalises the boundary, so the minimum x/y are 0/0.
	# Expects the complex form of the boundary.
	def normalise_boundary(self, boundary):
		min_i	= sys.float_info.max
		min_j	= sys.float_info.max

		for position in boundary:
			if position.real < min_i:
				min_i	= position.real
			if position.imag < min_j:
				min_j	= position.imag

		for index, val in enumerate(boundary):
			boundary[index]	= (val.real - min_i) + 1j * (val.imag - min_j)

		return boundary

	# Turns the contours provided by OpenCV into a numpy complex array.
	def contour_to_complex(self, contours, layer = 0):
		# [layer] strips out the array we care about.
		# Advanced indexing in numpy: [:, 0]
		# : gets ALL 'rows'.
		# 0 grabs the first element in each 'row'.
		contour	= contours[layer][:, 0]

		# Creates an empty np struct.
		# shape gives (len, 1, 2), i.e. an array of pairs length len.
		# [:-1] gives an array of elements length len.
		contour_complex			= np.empty(contour.shape[:-1], dtype = complex)
		contour_complex.real	= contour[:, 0]
		# Negated as OpenCV flips the y-axes normally, eases visualisation.
		contour_complex.imag	= -contour[:, 1]

		return contour_complex

	# Gets the lowest X% of frequency values from the fourier values.
	# Places back into the correct order.
	def get_low_frequencies_percentage(self, fourier_val, percent_to_keep):
		to_get		= int(len(fourier_val) * (percent_to_keep / 100))

		return self.get_low_frequencies(fourier_val, to_get)

	# Gets the lowest X of frequency values from the fourier values.
	# Places back into the correct order.
	def get_low_frequencies(self, fourier_val, to_get):
		fourier_freq		= np.fft.fftfreq(len(fourier_val))

		frequency_indices	= []
		for index, val in enumerate(fourier_freq):
			frequency_indices.append([index, val])

		# Sorts on absolute value of frequency (want negative and positive).
		frequency_indices.sort(key = lambda tuple: abs(tuple[1]))

		raw_values	= []
		for i in range(0, to_get):
			index	= frequency_indices[i][0]
			raw_values.append([fourier_val[index], index])

		# Sort based on original ordering.
		raw_values.sort(key = lambda tuple: tuple[1])

		# Strip out indices used for sorting.
		values	= []
		for value in raw_values:
			values.append(value[0])

		return values

	# Performs the inverse DFT.
	# If a subset of the original values are inverted, it scales the image.
	# Therefore scales back to original values.
	def inverse_fourier_and_scale(self, fourier_val, percentage = 100):
		inverted	= np.fft.ifft(fourier_val)
		inverted	= inverted / (100 / percentage)

		return inverted

	def get_shape_difference(self, img_dir, img_a, img_b, contour_level_a, contour_level_b):
		# Get contour represented as complex numbers
		contour_a		= self.get_complex_contour(img_dir, img_a, contour_level_a)
		contour_b		= self.get_complex_contour(img_dir, img_b, contour_level_b)

		# Get fourier descriptor from contours
		fourier_a		= np.fft.fft(contour_a)
		fourier_b		= np.fft.fft(contour_b)

		# Make result invariant to scale, translation, etc.
		fourier_a 	= self.make_scale_rotation_sp_invariant(fourier_a)
		fourier_b 	= self.make_scale_rotation_sp_invariant(fourier_b)

		# Uses a subset of the fourier coefficient as we only care
		# about the global form of the contours, not the details
		trun_a			= self.get_low_frequencies(fourier_a, 5)
		trun_b			= self.get_low_frequencies(fourier_b, 5)

		# make translation invariant
		final_a 		= self.make_translation_invariant(trun_a)
		final_b			= self.make_translation_invariant(trun_b)

		dist = distance.euclidean(final_a, final_b)

		img_a = self.read_image(img_dir, img_a)
		img_b = self.read_image(img_dir, img_b)

		# Generate opencv shape matching (uses Hu moments) for comparison
		ocv_contour_a = self.find_contours(self.get_copy(img_a))[contour_level_a]
		ocv_contour_b = self.find_contours(self.get_copy(img_b))[contour_level_b]

		opencv_dist = cv2.matchShapes(ocv_contour_a, ocv_contour_b, 1, 0)

		print "Distance generated from opencv.matchShapes: " + str(opencv_dist)
		self.display_shape_difference(img_a, img_b, dist)

	def display_shape_difference(self, img_a, img_b, distance):
		plt.subplot(1, 2, 1)
		plt.imshow(img_a, cmap = 'gray')
		plt.xticks([]), plt.yticks([])
		plt.title('Shape to Match: ')

		plt.subplot(1, 2, 2)
		plt.imshow(img_b, cmap = 'gray')
		plt.xticks([]), plt.yticks([])
		plt.title('Difference to Original Shape: \n' + str(np.round(distance, 3)))

		plt.show()

	# Returns a fourier descriptor that is invariant to scale, rotation and displacement (difference in starting point of shape)
	def make_scale_rotation_sp_invariant(self, fourier_desc):
		first_val = fourier_desc[0]

		for index, value in enumerate(fourier_desc):
		  fourier_desc[index] = np.absolute(value)/first_val

		return fourier_desc

	def make_translation_invariant(self, fourier_desc):
		return fourier_desc[1:len(fourier_desc)]

	# Plots the boundary.
	# Expects it as a numpy complex array.
	def plot_boundary(self, image, boundary, boundary_percent):
		dimensions	= np.shape(image)
		x_max		= dimensions[1]
		y_min		= -dimensions[0]

		plt.plot(boundary.real, boundary.imag, 'k')
		plt.xlim(0, x_max)
		plt.ylim(y_min, 0)
		plt.xticks([])
		plt.yticks([])
		plt.title('Truncated Boundary: ' + str(boundary_percent) + '%')
		plt.show()

	# Plots the boundary against the original image and boundary.
	# Expects boundaries as a numpy complex array.
	def plot_boundaries_and_image(self, image, image_name, original_boundary, new_boundaries, boundary_percentages, threshold_img):
		dimensions	= np.shape(image)
		x_max		= dimensions[1]
		y_min		= -dimensions[0]

		size		= 3 + len(new_boundaries)
		rows		= math.ceil(math.sqrt(size))
		columns		= math.ceil(size / rows)
		
		plt.subplot(rows, columns, 1)
		plt.imshow(image, cmap = 'gray')
		plt.xticks([]), plt.yticks([])
		plt.title('Source Image: ' + str(image_name))

		plt.subplot(rows, columns, 2)
		plt.imshow(threshold_img, cmap = 'gray')
		plt.xticks([]), plt.yticks([])
		plt.title('Threshold Image')

		plt.subplot(rows, columns, 3)
		plt.plot(original_boundary.real, original_boundary.imag, 'k')
		plt.xlim(0, x_max)
		plt.ylim(y_min, 0)
		plt.xticks([])
		plt.yticks([])
		plt.title('Original Boundary')

		for index, boundary in enumerate(new_boundaries): 
			plt.subplot(rows, columns, (4 + index))
			plt.plot(boundary.real, boundary.imag, 'k')
			plt.xlim(0, x_max)
			plt.ylim(y_min, 0)
			plt.xticks([]), plt.yticks([])
			plt.title('Truncated Boundary: ' + str(boundary_percentages[index]) + '%')

		plt.show()

	# Function for demonstration used to display truncated contours.
	def demo(self, image, image_name, is_img_rgb, percent_to_keep, contour_level):
		threshold_img	= self.apply_thresholding(image, is_img_rgb)
		contours		= self.find_contours(threshold_img)
		contour_complex	= self.contour_to_complex(contours, contour_level)
		fourier_val		= np.fft.fft(contour_complex)
		print fourier_val
		inverted		= []
		if percent_to_keep == 'all':
			percentages	= [1, 2, 3, 5, 10, 25, 50, 75, 100]
			for index, percent in enumerate(percentages):
				fourier_subset	= self.get_low_frequencies_percentage(fourier_val, percent)
				inverted.append(self.inverse_fourier_and_scale(fourier_subset, percent))
			self.plot_boundaries_and_image(image, image_name, contour_complex, inverted, percentages, threshold_img)
		else:
			fourier_subset	= self.get_low_frequencies_percentage(fourier_val, percent_to_keep)
			inverted.append(self.inverse_fourier_and_scale(fourier_subset, percent_to_keep))
			self.plot_boundaries_and_image(image, image_name, contour_complex, inverted, [percent_to_keep], threshold_img)

	# Provided with the image location and the contour_level required will
	# call the functions necessary and return the contour in its complex form.
	def get_complex_contour(self, img_dir, img_name, contour_level):
		img_orig		= self.read_image(img_dir, img_name)
		threshold_img	= self.apply_thresholding(img_orig, 0)
		contours		= self.find_contours(threshold_img)
		contour_complex	= self.contour_to_complex(contours, contour_level)
		return contour_complex
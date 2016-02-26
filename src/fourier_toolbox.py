# Future imports.
from __future__ import division

# External libraries.
import copy
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.widgets import Slider, CheckButtons
from scipy.spatial import distance

class fourier_toolbox:

	# Global for plotting to see.
	slider					= None
	tick_btn				= None
	plot_title_obj			= None
	visualisation_a_plot	= None
	visualisation_a_line	= None
	visualisation_b_plot	= None
	visualisation_b_line	= None

	img_dir					= None
	img_a					= None
	img_b					= None
	contour_level_a			= None
	contour_level_b			= None

	translation_inv			= False
	scale_inv				= False
	rotation_sp_inv			= False

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
	# Most exhaustive parameters fed in: CHAIN_APPROX_NONE
	# CHAIN_APPROX_SIMPLE appears faster, but less accurate (as expected).
	def find_contours(self, image):
		img_clone	= self.get_copy(image)

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

	def display_shape_difference(self, img_dir, img_a, img_b, contour_level_a, contour_level_b):
		default_frequencies	= 5

		self.img_dir			= img_dir
		self.img_a				= img_a
		self.img_b				= img_b
		self.contour_level_a	= contour_level_a
		self.contour_level_b	= contour_level_b

		dist, opencv_dist, boundary_a, boundary_b	= self.get_shape_difference(img_dir, img_a, img_b, contour_level_a, contour_level_b, default_frequencies)

		cv_img_a		= self.read_image(img_dir, img_a)
		cv_img_b		= self.read_image(img_dir, img_b)

		dimensions_a	= np.shape(cv_img_a)
		x_max_a			= dimensions_a[1]
		y_min_a			= -dimensions_a[0]

		dimensions_b	= np.shape(cv_img_b)
		x_max_b			= dimensions_b[1]
		y_min_b			= -dimensions_b[0]

		slider_axes		= plt.axes([0.2, 0.025, 0.6, 0.04])
		self.slider		= Slider(slider_axes, 'Fourier Descriptors', 3, 30, valinit = default_frequencies)
		self.slider.on_changed(self.update_slider)

		tick_btn_axes	= plt.axes([0.2, 0.9, 0.2, 0.1])
		tick_btn_axes.patch.set_visible(False)
		tick_btn_axes.axis('off')
		self.tick_btn	= CheckButtons(tick_btn_axes, ('Translation Invariance', 'Scale Invariance', 'Rotation Invariance'), (self.translation_inv, self.scale_inv, self.rotation_sp_inv))
		self.tick_btn.on_clicked(self.update_tickbox)

		plt.subplot(2, 2, 1)
		plt.imshow(cv_img_a, cmap = 'gray')
		plt.xticks([]), plt.yticks([])
		plt.title('Shape to Match: ')

		plt.subplot(2, 2, 2)
		plt.imshow(cv_img_b, cmap = 'gray')
		plt.xticks([]), plt.yticks([])
		self.plot_title_obj	= plt.title('Difference to Original Shape: \n' + str(np.round(dist, 3)))

		self.visualisation_a_plot	= plt.subplot(2, 2, 3)
		self.visualisation_a_line,	= plt.plot(boundary_a.real, boundary_a.imag, 'k')
		x_lims	= plt.xlim()
		x_delta	= x_lims[1] - x_lims[0]
		new_x	= [x_lims[0] - x_delta, x_lims[1] + x_delta]
		plt.xlim(new_x)
		y_lims	= plt.ylim()
		y_delta	= y_lims[1] - y_lims[0]
		new_y	= [y_lims[0] - y_delta, y_lims[1] + y_delta]
		plt.ylim(new_y)
		plt.xticks([])
		plt.yticks([])
		plt.title('Visualisation of Original\nShape\'s Descriptors')

		self.visualisation_b_plot	= plt.subplot(2, 2, 4)
		self.visualisation_b_line,	= plt.plot(boundary_b.real, boundary_b.imag, 'k')
		plt.xlim(new_x)
		plt.ylim(new_y)
		plt.xticks([])
		plt.yticks([])
		plt.title('Visualisation of Comparison\nShape\'s Descriptors')

		plt.show()

	def update_slider(self, value):
		no_frequencies	= int(round(self.slider.val))
		dist, opencv_dist, boundary_a, boundary_b	= self.get_shape_difference(self.img_dir, self.img_a, self.img_b, self.contour_level_a, self.contour_level_b, no_frequencies)

		new_title		= 'Difference to Original Shape: \n' + str(np.round(dist, 3))
		plt.setp(self.plot_title_obj, text = new_title)

		a_x_max	= np.amax(boundary_a.real)
		a_x_min	= np.amin(boundary_a.real)
		a_x_del	= (a_x_max - a_x_min) / 3

		a_y_max	= np.amax(boundary_a.imag)
		a_y_min	= np.amin(boundary_a.imag)
		a_y_del	= (a_y_max - a_y_min) / 3

		b_x_max	= np.amax(boundary_b.real)
		b_x_min	= np.amin(boundary_b.real)
		b_x_del	= (b_x_max - b_x_min) / 3

		b_y_max	= np.amax(boundary_b.imag)
		b_y_min	= np.amin(boundary_b.imag)
		b_y_del	= (b_y_max - b_y_min) / 3

		self.visualisation_a_line.set_data(boundary_a.real, boundary_a.imag)
		self.visualisation_a_plot.set_xlim([a_x_min - a_x_del, a_x_max + a_x_del])
		self.visualisation_a_plot.set_ylim([a_y_min - a_y_del, a_y_max + a_y_del])

		self.visualisation_b_line.set_data(boundary_b.real, boundary_b.imag)
		self.visualisation_b_plot.set_xlim([b_x_min - b_x_del, b_x_max + b_x_del])
		self.visualisation_b_plot.set_ylim([b_y_min - b_y_del, b_y_max + b_y_del])

		plt.draw()

	def update_tickbox(self, label):
		if label == 'Translation Invariance':
			self.translation_inv	= not self.translation_inv
		elif label == 'Scale Invariance':
			self.scale_inv			= not self.scale_inv
		elif label == 'Rotation Invariance':
			self.rotation_sp_inv	= not self.rotation_sp_inv

		# Bit of a hack, but hey, code re-use.
		self.update_slider(0)


	def get_shape_difference(self, img_dir, img_a, img_b, contour_level_a, contour_level_b, no_frequencies):
		# Get contour represented as complex numbers
		contour_a		= self.get_complex_contour(img_dir, img_a, contour_level_a)
		contour_b		= self.get_complex_contour(img_dir, img_b, contour_level_b)

		# Get fourier descriptor from contours
		fourier_a		= np.fft.fft(contour_a)
		fourier_b		= np.fft.fft(contour_b)

		# Make rotation and starting point invariant
		if self.rotation_sp_inv:
			fourier_a	= self.make_rotation_sp_invariant(fourier_a)
			fourier_b	= self.make_rotation_sp_invariant(fourier_b)

		# Make scale invariant
		if self.scale_inv:
			fourier_a 	= self.make_scale_invariant(fourier_a)
			fourier_b 	= self.make_scale_invariant(fourier_b)

		# Make translation invariant
		if self.translation_inv:
			fourier_a 	= self.make_translation_invariant(fourier_a)
			fourier_b	= self.make_translation_invariant(fourier_b)

		# Uses a subset of the fourier coefficient as we only care
		# about the global form of the contours, not the details
		fourier_a	= self.get_low_frequencies(fourier_a, no_frequencies)
		fourier_b	= self.get_low_frequencies(fourier_b, no_frequencies)

		dist = distance.euclidean(fourier_a, fourier_b)

		img_a = self.read_image(img_dir, img_a)
		img_b = self.read_image(img_dir, img_b)

		# Generate opencv shape matching (uses Hu moments) for comparison
		ocv_contour_a = self.find_contours(self.get_copy(img_a))[contour_level_a]
		ocv_contour_b = self.find_contours(self.get_copy(img_b))[contour_level_b]

		# Calculate the size of the truncation compared to original
		# for inverse_fourier_and_scale(), as a percent of the original.
		fourier_a_percent	= 100 * (len(fourier_a) / len(contour_a))
		fourier_b_percent	= 100 * (len(fourier_b) / len(contour_b))

		opencv_dist = cv2.matchShapes(ocv_contour_a, ocv_contour_b, 1, 0)

		return dist, opencv_dist, self.inverse_fourier_and_scale(fourier_a, fourier_a_percent), self.inverse_fourier_and_scale(fourier_b, fourier_b_percent)

	# Returns a fourier descriptor that is invariant to rotation and boundary starting point.
	def make_rotation_sp_invariant(self, fourier_desc):
		for index, value in enumerate(fourier_desc):
		  fourier_desc[index] = np.absolute(value)

		return fourier_desc

	# Returns a fourier descriptor that is invariant to scale.
	def make_scale_invariant(self, fourier_desc):
		first_val	= fourier_desc[0]

		for index, value in enumerate(fourier_desc):
			fourier_desc[index]	= value / first_val

		return fourier_desc

	# Returns a fourier descriptor that is invariant to translation.
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
# Expected command line inputs:
# 1) Image name (residing in images/).
# 2) Whether the image is RGB: 0 = false, 1 = true.
# 3) Percentage of the Fourier values to utilise in inversion: 1->100.
# 4) Contour level of the image (may be several, generally the one you want
# has to be found via trial and error on the image if there are several): 0->X.
# E.g. python testing.py star.jpg 0 10 0

# Future imports.
from __future__ import division

# External libraries.
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pprint
import sys

# Image directory.
img_dir	= '../images/'

# Parse command line arguments and read in the image.
args			= sys.argv
img				= args[1]
is_rgb			= int(args[2]) == 1
percentage		= int(args[3])
contour_level	= int(args[4])
img_orig		= cv2.imread(img_dir + img, 0)

# Converts an RGB image into greyscale.
# Then converts the greyscale image into a binary one, and returns the threshold image.
def clean(image, src_rgb):
	img_clone	= fresh_image()

	if src_rgb:
		img_clone	= cv2.cvtColor(img_clone, cv2.COLOR_BGR2GRAY)

	ret, threshold	= cv2.threshold(img_clone, 127, 255, cv2.THRESH_BINARY)

	return threshold

# Finds and returns the contours.
# TODO: Expand to other parameters:
# Most exhaustive parameters fed in, change later if possible - specifically CHAIN_APPROX_NONE to CHAIN_APPROX_SIMPLE.
def find_contours(image):
	img_clone	= fresh_image(image)

	contours, hierarchy	= cv2.findContours(img_clone, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	return contours

# Basic image display for a single image.
# Closes all images after use.
def display_image(image, title):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Returns a clone of an image, or the original image if no argument passed.
def fresh_image(image = img_orig):
	return copy.deepcopy(image)

# Normalises the boundary, so the minimum x/y are 0/0.
# Expects the complex form of the boundary.
def normalise_boundary(boundary):
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
def contour_to_complex(contours, layer = 0):
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
def get_low_frequencies_percentage(fourier_val, percentage):
	fourier_freq		= np.fft.fftfreq(len(fourier_val))

	frequency_indices	= []
	for index, val in enumerate(fourier_freq):
		frequency_indices.append([index, val])

	# Sorts on absolute value of frequency (want negative and positive).
	frequency_indices.sort(key = lambda tuple: abs(tuple[1]))

	to_get		= int(len(frequency_indices) * (percentage / 100))

	raw_values	= []
	for i in range(0, to_get):
		index	= frequency_indices[i][0]
		raw_values.append([fourier_val[index], index])

	# Sort based on original ordering.
	raw_values.sort(key = lambda tuple: tuple[1])
	# Strip out indices used for sorting.
	#values	= values[:][0]

	# Strip out indices used for sorting.
	values	= []
	for value in raw_values:
		values.append(value[0])

	return values

# Performs the inverse DFT.
# If a subset of the original values are inverted, it scales the image.
# Therefore scales back to original values.
def inverse_fourier_and_scale(fourier_val, percentage = 100):
	inverted	= np.fft.ifft(fourier_val)
	inverted	= inverted / (100 / percentage)

	return inverted

# Plots the boundary.
# Expects it as a numpy complex array.
def plot_boundary(boundary, img_orig):
	dimensions	= np.shape(img_orig)
	x_max		= dimensions[1]
	y_min		= -dimensions[0]

	plt.plot(boundary.real, boundary.imag)
	plt.xlim(0, x_max)
	plt.ylim(y_min, 0)
	plt.show()

# Plots the boundary against the original image and boundary.
# Expects boundaries as a numpy complex array.
def plot_boundaries_and_image(original_boundary, new_boundary, img_orig, threshold_img):
	dimensions	= np.shape(img_orig)
	x_max		= dimensions[1]
	y_min		= -dimensions[0]
	
	plt.subplot(2, 2, 1)
	plt.imshow(img_orig, cmap = 'gray')
	plt.xticks([]), plt.yticks([])
	plt.title('Source Image: ' + str(img))

	plt.subplot(2, 2, 2)
	plt.imshow(threshold_img, cmap = 'gray')
	plt.xticks([]), plt.yticks([])
	plt.title('Threshold Image')

	plt.subplot(2, 2, 3)
	plt.plot(original_boundary.real, original_boundary.imag, 'k')
	plt.xlim(0, x_max)
	plt.ylim(y_min, 0)
	plt.xticks([]), plt.yticks([])
	plt.title('Original Boundary')

	plt.subplot(2, 2, 4)
	plt.plot(new_boundary.real, new_boundary.imag, 'k')
	plt.xlim(0, x_max)
	plt.ylim(y_min, 0)
	plt.xticks([]), plt.yticks([])
	plt.title('Truncated Bondary')

	plt.show()

threshold_img	= clean(img_orig, is_rgb)
contours		= find_contours(threshold_img)
contour_complex	= contour_to_complex(contours, contour_level)
fourier_val		= np.fft.fft(contour_complex)
fourier_subset	= get_low_frequencies_percentage(fourier_val, percentage)
inverted		= inverse_fourier_and_scale(fourier_subset, percentage)
plot_boundaries_and_image(contour_complex, inverted, img_orig, threshold_img)
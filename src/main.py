# Expected command line inputs for boundary truncation demonstration:
# 1) Run mode: 'boundary_demo'.
# 2) Image name (residing in images/) (including file extension).
# 3) Whether the image is RGB: 0 = false, 1 = true.
# 4) Percentage of the Fourier values to utilise in inversion: 1->100.
# Can also be 'all' which will result in a far more verbose output comparing several percentages.
# 5) Contour level of the image (may be several, generally the one you want
# has to be found via trial and error on the image if there are several): 0->X.
# E.g. python main.py boundary_demo star.jpg 0 10 0
# python main.py boundary_demo penguin.jpg 0 all 1

# External libraries.
# TODO: Remove unused later.
import cv2
import sys
import math
import numpy as np

from scipy.spatial import distance

# Internal classes.
from fourier_toolbox import fourier_toolbox

# Image directory.
img_dir	= '../images/'
'''
# Parse command line arguments and read in the image.
args			= sys.argv
run_mode		= args[1]
if run_mode == 'boundary_demo':
	img_name		= args[2]
	is_img_rgb		= int(args[3]) == 1
	percent_to_keep	= args[4]
	if percent_to_keep != 'all':
		percent_to_keep	= int(percent_to_keep)
	contour_level	= int(args[5])

if run_mode == 'boundary_demo':
	fourier_toolbox	= fourier_toolbox()
	img_orig		= fourier_toolbox.read_image(img_dir, img_name)
	fourier_toolbox.demo(img_orig, img_name, is_img_rgb, percent_to_keep, contour_level)
'''

# This'll be used for comparing with moment methods.
img_a	= 'penguin.jpg'
img_b	= 'penguin_rotated.jpg'

toolbox	= fourier_toolbox()

contour_a	= toolbox.get_complex_contour(img_dir, img_a, 1)
contour_b	= toolbox.get_complex_contour(img_dir, img_b, 1)

#for index, value in enumerate(contour_b):
#	if value != contour_a[index]:
#		print math.abs(value - contour_a[index])

fourier_a	= np.fft.fft(contour_a)
fourier_b	= np.fft.fft(contour_b)

trun_a		= toolbox.get_low_frequencies(fourier_a, 10)
trun_b		= toolbox.get_low_frequencies(fourier_b, 10)

# Translation invariance.


print distance.euclidean(trun_a, trun_b)

#print np.linalg.norm(contour_a - contour_b)

#print len(contour_a)
#print len(contour_b)
'''
print cv2.matchShapes(contour_a, contour_b, 1, 0)
'''

# Remove DC from getlowfreq / somewhere else.
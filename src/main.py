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

# Internal classes.
from fourier_toolbox import fourier_toolbox

# Image directory.
img_dir	= '../images/'

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

# This'll be used for comparing with moment methods.
'''
img_a	= 'penguin.jpg'
img_b	= 'star.jpg'

toolbox_a	= fourier_toolbox(img_dir, img_a)
toolbox_b	= fourier_toolbox(img_dir, img_b)

contour_a	= toolbox_a.test(1)
contour_b	= toolbox_b.test(0)

print cv2.matchShapes(contour_a, contour_b, 1, 0)
'''
# Expected command line inputs for boundary truncation demonstration:
# 1) Run mode: 'boundary_demo'.
# 2) Image name (residing in images/) (including file extension).
# 3) Whether the image is RGB: 0 = false, 1 = true.
# 4) Percentage of the Fourier values to utilise in inversion: 1->100.
# Can also be 'all' which will result in a far more verbose output comparing several percentages.
# 5) Contour level of the image (may be several, generally the one you want
# has to be found via trial and error on the image if there are several): 0->X.
# E.g. python main.py boundary_demo star.jpg 0 10 0

# Expected command line inputs for image comparison demonstration:
# 1) Run mode: 'compare_images'.
# 2) Image A name (residing in images/) (including file extension).
# 3) Image B name (as above).
# 3) Contour level of image A (may be several, generally the one you want
# has to be found via trial and error on the image if there are several): 0->X.
# 4) Contour level of image B (as above).
# 5) Number of frequencies to keep for comparison (note this is N+1 when translation
# invariance is desired, as it also removes the DC-term).
# E.g. python main.py compare_images kitty.jpg kitty_rotated.jpg 95 95 10

# External libraries.
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
elif run_mode == 'compare_images':
	img_name_a			= args[2]
	img_name_b			= args[3]
	img_a_contour_level	= int(args[4])
	img_b_contour_level	= int(args[5])
	no_frequencies		= int(args[6])

fourier_toolbox	= fourier_toolbox()
if run_mode == 'boundary_demo':
	img_orig		= fourier_toolbox.read_image(img_dir, img_name)
	fourier_toolbox.demo(img_orig, img_name, is_img_rgb, percent_to_keep, contour_level)
elif run_mode == 'compare_images':
	fourier_toolbox.get_shape_difference(img_dir, img_name_a, img_name_b, img_a_contour_level, img_b_contour_level, no_frequencies)
# External libraries.
import sys

# Internal classes.
from fourier_toolbox import fourier_toolbox

# Image directory.
img_dir	= '../images/'

# Parse command line arguments and read in the image.
args			= sys.argv
img_name		= args[1]
is_img_rgb		= int(args[2]) == 1
percent_to_keep	= args[3]
if percent_to_keep != 'all':
	percent_to_keep	= int(percent_to_keep)
contour_level	= int(args[4])

fourier_toolbox	= fourier_toolbox(img_dir, img_name)
fourier_toolbox.demo(is_img_rgb, percent_to_keep, contour_level)
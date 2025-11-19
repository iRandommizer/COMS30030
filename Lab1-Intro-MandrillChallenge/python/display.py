################################################
#
# COMS30068 - display.py
# TOPIC: create, save and display an image
#
# Getting-Started-File for OpenCV
# University of Bristol
#
################################################

import cv2
import numpy as np

# Get the image info -------------------
# load image from a file into the container
image = cv2.imread("myimage.jpg", cv2.IMREAD_UNCHANGED) # - Reads an exsiting file
# --------------------------------------

# Create ans show Image ----------------
# construct a window for image display
namedWindow = 'Display window'

# visualise the loaded image in the window
cv2.imshow(namedWindow, image)
# --------------------------------------

# wait for a key press until returning from the programw
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
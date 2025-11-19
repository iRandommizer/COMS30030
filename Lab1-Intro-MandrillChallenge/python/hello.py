################################################
#
# COMS30068 - hello.py
# TOPIC: create, save and display an image
#
# Getting-Started-File for OpenCV
# University of Bristol
#
################################################

import cv2
import numpy as np

# create a black 256x256, 8bit, gray scale image in a matrix container
image = np.zeros([256,400, 3], dtype=np.uint8)

# drawwhite text HelloOpenCV!
image = cv2.putText(
    image,                          # The image to draw texts
    "Welcome to the Matrix!",       # The text string to write
    (70, 70),                       # Bottom-left corner coordinates where the text starts (x = 70, y = 70)
    cv2.FONT_HERSHEY_COMPLEX_SMALL, # The font type used for the text
    0.8,                            # Font scale (size)
    (98, 160, 3),                   # Text Color in BGR format
    1,                              # Thickness of the text strokes
    cv2.LINE_AA)                    # Line type for anti-aliased (smooth) text

# save image to file
cv2.imwrite("myimage.jpg", image) # imwrite allows you to save the file

# construct a window for image display
namedWindow = 'Display window'

# visualise the loaded image in the window
cv2.imshow(namedWindow, image)

# wait for a key press until returning from the programw
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
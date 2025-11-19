# Task 1 - Convolution
# PseudoCode
# For every pixel, 
#   get all 8 adjacent pixels + cur pix's brightness value 
#   if edge, only take applicable pixels and divide accoridngly
 
import cv2 # Old OpenCV interface
import numpy as np

image = cv2.imread("../images/mandrill.jpg",0)
ref = cv2.imread("../images/mandrill.jpg",0)

def get_3x3_sqr_value(y_loc, x_loc, img):
    neighbour_offset = [(1,-1),(1,0),(1,1),(0,-1),(0,0),(0,1),(-1,-1),(-1,0),(-1,1)]
    valid_space = [] # instantiate the container
    for ny, nx in neighbour_offset: # for 
        delta_y_loc, delta_x_loc = y_loc + ny, x_loc + nx
        if delta_y_loc >= 0 and delta_y_loc < img.shape[0] and delta_x_loc >= 0 and delta_x_loc < img.shape[1]:
            valid_space.append((delta_y_loc,delta_x_loc))
    return valid_space

for y in range(0,int(image.shape[0]/2)):
    for x in range(0, int(image.shape[1]/2)):
        list_of_valid_space = get_3x3_sqr_value(y,x,ref)
        brightness_val = 0
        for val in list_of_valid_space:
            brightness_val = brightness_val + int(ref[val])
        image[y,x] = int(brightness_val/len(list_of_valid_space))

# Let's say y,x = 1,1 hence we need:
# 1,1 | 1,2 | 2,1 | 2,2

namedWindow = "Display Window"

cv2.imshow(namedWindow, image)

cv2.waitKey(0)

cv2.destroyAllWindows()
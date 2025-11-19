import cv2
import numpy as np
import math

image = cv2.imread("../images/car2.png", 0)
filtered_image = image.copy()

# Helper Function to help me get the offsets for the neighbours
def create_loc_mapping(size):
    loc_array = []
    starting_pos = size//2
    # This order of nesting produces a left to right, bottom to top order
    for y in range(0,size):
        for x in range(0,size):
            loc_array.append((x - starting_pos, y - starting_pos))
    return loc_array
    # Returns: [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]

def check_bounds(input_img, dx, dy):
    return (dx >= 0 and dx < input_img.shape[0] and dy >= 0 and dy < input_img.shape[1])

def median_filter_function(input_img,size):
    img_copy = input_img.copy()
    # For every pixel starting from Top Left, left to right, bottom to top:
    for y in range(img_copy.shape[1]):
        for x in range(img_copy.shape[0]):
            neighbour_vals = []
            # Get it's neighbouring brightness values in a list
            offset_loc = create_loc_mapping(size)
            # For every offset location
            for offset in offset_loc:
                new_x = x + offset[0]
                new_y = y + offset[1]
                # Make sure it's inside the bounds
                if check_bounds(img_copy, new_x, new_y):
                    # Get the brightness value of that neighbour
                    neighbour_vals.append(img_copy[new_x,new_y])
            neighbour_vals.sort()
            input_img[x,y] = neighbour_vals[len(neighbour_vals)//2]

median_filter_function(filtered_image,5)

cv2.imshow("Origina", image)
cv2.imshow("Filtered", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

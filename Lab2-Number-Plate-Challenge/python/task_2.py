import cv2 
import numpy as np
import math

image = cv2.imread("../images/car1.png", 1)
blurred = image.copy()

# create gaussian blur returns a 2 dimension array which is a gausian blur
# this just gives us the weights influence on how our valid space map
def gaussian_function_mapping(size, sigma):
    # size - var for determining the kernel dimensions the guassian fn
    # sigma - gaussian spread (std dev)
    array = np.zeros((size,size))
    center = size // 2
    total = 0
    for y in range(size):
        for x in range(size):
            dy = y - center
            dx = x - center
            exponent = -((dx**2 + dy**2)/ (2 * sigma ** 2) ) # sigma refers to the standard deviation value
            array[y,x] = 1 / (2 * math.pi * (sigma ** 2)) * math.exp(exponent)
            total += array[y,x]
    array = array / total
    return array 

def create_loc_mapping(size):
    loc_array = []
    starting_pos = size//2
    for y in range(0,size):
        for x in range(0,size):
            loc_array.append((x - starting_pos, y - starting_pos))
    return loc_array

# I made 2 issues
# 1)I used input image instead of img_copy for checking the adjacent pixels
# 2) When I was trying to acess the data of the kernel, i was giving it -2,-2 instead of values range from 0 to 4
def process_image(input_img, kernel=None, loc_mapping=None):
    img_copy = input_img.copy()
    center = kernel.shape[0] // 2
    
    for y in range(img_copy.shape[0]):
        for x in range(img_copy.shape[1]):
            compiled_pxl_value = np.zeros(3)
            
            for ox, oy in loc_mapping:
                dx, dy = x + ox, y + oy
                if dy >= 0 and dy < img_copy.shape[0] and dx >= 0 and dx < img_copy.shape[1]:
                    kernel_y = oy + center
                    kernel_x = ox + center
                    compiled_pxl_value += (img_copy[dy, dx] * kernel[kernel_y][kernel_x])
            
            input_img[y, x] = compiled_pxl_value
            
size = 7
sigma = size/2              
process_image(blurred, gaussian_function_mapping(size, sigma), create_loc_mapping(size))

detail = cv2.subtract(image, blurred)
alpha = 3
sharpened = cv2.addWeighted(image, 1.0, detail, alpha, 0) 

cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred)
cv2.imshow("Details - High Frequency", detail)
cv2.imshow("Sharper" , sharpened)

cv2.waitKey(0)

cv2.destroyAllWindows()
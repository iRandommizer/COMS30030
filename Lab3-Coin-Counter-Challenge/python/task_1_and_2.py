import cv2 
import numpy as np

image1 = cv2.imread("../images/coins1.png", 0)

# Detects vertical edges
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
# Detects horizontal edges
sobel_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])

# From Wk2 Task_2.py
def create_loc_mapping(size):
    loc_array = []
    starting_pos = size//2
    for y in range(0,size):
        for x in range(0,size):
            loc_array.append((x - starting_pos, y - starting_pos))
    return loc_array
    # Returns: [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]

def check_bounds(input_img, x, y):
    # x = column index → check against shape[1] (width)
    # y = row index → check against shape[0] (height)
    return (x >= 0 and x < input_img.shape[1] and 
            y >= 0 and y < input_img.shape[0])

# The function is suppose to find edges(usring the sharp brightness changes) in an image by detecting gradients
def sobel_operation(input_img, mapping_loc_array):
    ref = input_img.copy()
    G_x = np.zeros_like(input_img, dtype=float)
    G_y = np.zeros_like(input_img, dtype=float)
    
    # when using numpy, we imagine things in rows and columns, so just take note of the shift in loop nesting
    # For every row of pixels in img
    for y in range(input_img.shape[0]):  # FIXED: shape[0] = rows
        # For every col of pixels in img
        for x in range(input_img.shape[1]):  # FIXED: shape[1] = cols
            
            # Sobel Value accumalated for each pixel
            cur_processed_val_x = 0
            cur_processed_val_y = 0
            
            # For every neighbour based on the mapping offset array            
            for offset in mapping_loc_array:
                # Get the new x and y values for the neighbors
                offset_x = offset[0]
                offset_y = offset[1]
                new_x = x + offset_x
                new_y = y + offset_y
                
                # if neighbour is valid
                if check_bounds(ref, new_x, new_y):

                    # Convert offset to kernel index for the sobel kernel
                    kernel_x = offset_x + 1  # -1,0,1 → 0,1,2
                    kernel_y = offset_y + 1  # -1,0,1 → 0,1,2
                    
                    # Calculate the pixel * by the Sobel Kernel (NOtes, transpose is not needed anymore because we are using, [y,x] isntead of [x,y])
                    cur_processed_val_x += ref[new_y, new_x] * sobel_x[kernel_y, kernel_x]
                    cur_processed_val_y += ref[new_y, new_x] * sobel_y[kernel_y, kernel_x]
            
            # Store accumulated values
            G_x[y, x] = cur_processed_val_x
            G_y[y, x] = cur_processed_val_y
    
    # Calculate magnitude
    magnitude = np.sqrt(G_x**2 + G_y**2)

    # Normalise the magnitude (its basically percentage * 255), eg. 50% * 255 = 123
    magnitude = (magnitude - magnitude.min())/(magnitude.max() - magnitude.min())*255

    input_img[:,:] = magnitude.astype(np.uint8)
    return G_x, G_y

def threshold_image(input_image, threshold_val):
    # for np.where: NumPy automatically broadcasts the single value to compare against every pixel
    input_image[:,:] = np.where(input_image > threshold_val, 255, 0)


G_x, G_y = sobel_operation(image1, create_loc_mapping(3))
threshold_image(image1, 50)

cv2.imshow("Edges", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()


def hough_transform_circle_detection(input_img, g_x, g_y, peak_threshold, min_radius, max_radius):
    r, c = input_img.shape
    # Create your accumilator space
    # For every row
    for row in range(r):
        # For every col
        for col in range(c):
            # Check if pixel value >= 0
            if input_img[]
                # Get gradient ang
                # Draw circle within the accumilato space relative from that pixel value position?
                    # When drawing, get the current accumilator space to be drawn on and add 1 to it,
                    # this allows the pixel value to keep increasing if it keeps being drawn on
    # Once all circles are drawn in accumilator space, "bin" your pixel space and for each bin,
    # see those with the max value, and add to list
    # Count total in the list, that's how many circles there are?

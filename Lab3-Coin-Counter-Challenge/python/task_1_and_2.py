import cv2 
import numpy as np

image1 = cv2.imread("../images/coins1.png", 0)
# We need the output image to be BGR so our circles has colours
output_img = cv2.cvtColor("../images/coins1.png", cv2.COLOR_GRAY2BGR)


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
    # array for x gradient vectors and y gradient vectors
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

    # Calculate direction of gradient, I understand in the slides, they dont ask us to use arctan2 but 
    # in game dev, we use arctan2 instead of arctan so that we get the full range of direction (-180 to 180)
    # instead of just (-90 to 90)
    G_dir = np.arctan2(G_y,G_x) 

    return G_x,G_y,magnitude, G_dir

def data_to_image(data):
    data = np.abs(data)
    noramlised_data = (data - data.min())/(data.max() - data.min())*255
    return noramlised_data.astype(np.uint8)

def threshold_image(input_image, threshold_val):
    # for np.where: NumPy automatically broadcasts the single value to compare against every pixel
    input_image[:,:] = np.where(input_image > threshold_val, 255, 0)

def hough_transform_circle_detection(input_img, gradient_img, peak_threshold, min_radius, max_radius):
    # Input:
    # - Input image of "Thresholded" magnitude image
    # - Gradient image, need to use the raw gradient values, instead of the normalised one
    # - Hough Transform Peak Threshold
    # - Min Radius
    # - Max Radius
    r, c = input_img.shape
    radius_range = max_radius - min_radius + 1 # we add +1 to ensure that the max radius is included 
    accumalator = np.zeros((r,c,radius_range))
    # Create your accumilator space
    # For every row
    for row in range(r):
        # For every col
        for col in range(c):
            # Check if pixel value > 0
            if input_img[row, col] > 0:
                theta = gradient_img[row, col]
                # Based on gradient direction, find a & b
                for r in range(min_radius, max_radius+1):
                    a_p = int(round(col + r * np.cos(theta))) # Need to round up to int
                    b_p = int(round(row + r * np.sin(theta)))
                    a_n = int(round(col - r * np.cos(theta)))
                    b_n = int(round(row - r * np.sin(theta)))
                    # Check if both points are within the image space bounds, if within image space,
                    # accumalte in hough space
                    if check_bounds(input_img, a_p, b_p):
                        accumalator[b_p, a_p, r - min_radius] += 1
                    if check_bounds(input_img, a_n, b_n):
                        accumalator[b_n, a_n, r - min_radius] += 1

    # Circle
    circle_idx = np.where(accumalator > peak_threshold)
    circles = []
    for b,a,r_idx in zip(*circle_idx):
        radius = r_idx + min_radius
        circles.append((a,b,radius))

    #Hough Space 2D
    hough_space_2d = np.sum(accumalator, axis=2)
    hough_space_2d_log = np.log1p(hough_space_2d)

    # Normalize to [0, 1], right now all the accumalator pixels are too bright and barely any contrast with one another
    hough_norm = (hough_space_2d_log - hough_space_2d_log.min()) / (hough_space_2d_log.max() - hough_space_2d_log.min())

    # Since our values are now normalised from 0 to 1, if we apply the power-law transformation, we supress the lows
    # and preserve the peaks.
    # The higher the gamma value, greater the contrast between the lows ad the peaks
    gamma = 5 
    hough_gamma = hough_norm ** gamma
    return circles, hough_gamma

def draw_deteccted_circles(input_img, circles):
    for 

G_x, G_y, magnitude, G_dir = sobel_operation(image1, create_loc_mapping(3))
G_x_img = data_to_image(G_x)
G_y_img = data_to_image(G_y)
G_dir_img = data_to_image(G_dir)
magnitude_img = data_to_image(magnitude)
threshold_image(magnitude_img, 50)
circles, hough_space = hough_transform_circle_detection(magnitude_img, G_dir, 50, 35, 55)

cv2.imshow("G_x", G_x_img)
cv2.imshow("G_y", G_y_img)
cv2.imshow("magnitude", magnitude_img)
cv2.imshow("G_dir", G_dir_img)
cv2.imshow("Hough Space", hough_space)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np

image = cv2.imread("../images/mandrill.jpg", cv2.IMREAD_GRAYSCALE)

namedWindow = "Display Window"

threshold_value = 10
threshold_value_2 = 50

for y in range(0, image.shape[0]):  # go through all rows (or scanlines)
    for x in range(0, image.shape[1]):  # go through all columns
        if image[y,x] > threshold_value:
            image[y,x] = 255
        elif image[y,x] <= threshold_value_2:
            image[y,x] = 0
cv2.imshow(namedWindow, image)

cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
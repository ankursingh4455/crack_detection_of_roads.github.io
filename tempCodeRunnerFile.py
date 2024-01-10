# library
import numpy as np
import cv2
from matplotlib import pyplot as plt

# desired input
img = cv2.imread('Input-Set/Cracked_01.jpg')

# greyscaling the image(blurring to reduce the size of image and make it easier to process)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Image processing
# Averaging(Takes the average arithematic mean of pixel intensities and enhances the noised image)
blur = cv2.blur(gray, (3, 3))

# Logarithmic(Enhancing the higher pixel values)
img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
img_log = np.array(img_log, dtype=np.uint8)

# bilateral filter(Intensity difference between the edges and neighbouring pixels)
bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

# Canny Edge Detection(Single pixel width edges)
edges = cv2.Canny(bilateral, 100, 200)

# Morphological Closing Operator(filling out shape)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Create feature detecting method
orb = cv2.ORB_create(nfeatures=1500)

# Keypoints and Descriptors(edges and keyfeatures of image is )
keypoints, descriptors = orb.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)

# Create an output image
cv2.imshow("Road Crack Detected", featuredImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

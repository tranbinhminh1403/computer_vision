import cv2
import numpy as np
from PIL import Image

img = cv2.imread('images/box.png', 0)

minHessian = 400
sift = cv2.SIFT_create(minHessian)

keypoints, descriptors = sift.detectAndCompute(img, None)

#-- Draw keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None)
cv2.imwrite('sift_keypoints.jpg', img_keypoints)

Image.open('sift_keypoints.jpg').show()

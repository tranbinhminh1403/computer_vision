import os
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import numpy as np
import imutils

imageA = cv2.imread('./panorama/sedona_left.png')
imageB = cv2.imread('./panorama/sedona_right.png')

plt.figure(figsize=(16,16))
plt.subplot(121), plt.imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
plt.show()

# stitcher = cv2.createStitcher('SCANS', False)
stitcher = cv2.Stitcher_create()
#stitcher = cv2.Stitcher.create()
result = stitcher.stitch((imageA, imageB))
stitched=result[1]
cv2.imwrite('sedona.png', result[1])

# Image('mountain1.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.show()

# create a 10 pixel border surrounding the stitched image
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,cv2.BORDER_CONSTANT, (0, 0, 0)) 

# convert the stitched image to grayscale and threshold it
# such that all pixels greater than zero are set to 255
# (foreground) while all others remain 0 (background)
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# find all external contours in the threshold image then find
# the *largest* contour which will be the contour/outline of
# the stitched image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
mask = np.zeros(thresh.shape, dtype="uint8")
(x, y, w, h) = cv2.boundingRect(c)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# create two copies of the mask: one to serve as our actual
# minimum rectangular region and another to serve as a counter
# for how many pixels need to be removed to form the minimum
# rectangular region
minRect = mask.copy()
sub = mask.copy()
while cv2.countNonZero(sub) > 0:
    # erode the minimum rectangular mask and then subtract
	# the thresholded image from the minimum rectangular mask
	# so we can count if there are any non-zero pixels left
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)
    
# find contours in the minimum rectangular mask and then
# extract the bounding box (x, y)-coordinates    
cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(c)

# use the bounding box coordinates to extract the our final
# stitched image
stitched = stitched[y:y + h, x:x + w]

# write the output stitched image to disk
cv2.imwrite('sedona_stic.png', stitched)

#Image('mountain1_stic.png')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.show()
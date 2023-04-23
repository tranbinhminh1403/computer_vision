import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.system('cls' if os.name == 'nt' else 'clear')

img = cv2.imread('images/deer_salt.jpg',cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,10))
plt.imshow(img,cmap='gray')
plt.show()

img = cv2.imread('images/deer_salt.jpg',cv2.IMREAD_GRAYSCALE)
filtered = cv2.medianBlur(img, 3) # apply median filter with kernel size of 3
plt.figure(figsize=(10,10))
plt.imshow(filtered,cmap='gray')
plt.show()
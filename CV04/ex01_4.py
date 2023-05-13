import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

def sobel_edge_detection(image_path, blur_ksize=7, sobel_ksize=1, skipping_threshold=30):
    ### YOUR CODE HERE ###
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    #sobel
    img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    img_sobely = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    img_sobel = np.sqrt(np.square(img_sobelx) + np.square(img_sobely))
    
    img_sobel = np.absolute(img_sobel)
    img_sobel = np.uint8(img_sobel)
    
    #Cắt ngưỡng 
    img_sobel[img_sobel < skipping_threshold] = 0
    img_sobel[img_sobel >= skipping_threshold] = 255
    
    return img_sobel

image_path = './lane_detection/lane1.png'

img_sobel = sobel_edge_detection(image_path, 3, 1, 30)

plt.figure(figsize=(12,12))
plt.imshow(img_sobel,cmap='gray')
plt.show()
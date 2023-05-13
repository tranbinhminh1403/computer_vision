import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

def canny_edge_detection(image_path, blur_ksize=5, threshold1=100, threshold2=200, skipping_threshold=30):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    img_canny = cv2.Canny(img_gaussian, threshold1, threshold2)
    
    img_canny[img_canny < skipping_threshold] = 0
    img_canny[img_canny >= skipping_threshold] = 255
    return img_canny

image_path = 'images/seed.png'
gray = cv2.imread(image_path, 0)
img_canny = canny_edge_detection(image_path, 7, 1, 30)

plt.subplot(121),plt.imshow(gray, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_canny,cmap = 'gray')
plt.title('Sobel Image'), plt.xticks([]), plt.yticks([])
plt.show()

image_dir = 'data/'
list_c = []
for file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, file)
    skipping_threshold = 30
    img_canny = canny_edge_detection(image_path, skipping_threshold=skipping_threshold)
    chuvi = 0
    for i in range(img_canny.shape[0]):
        for j in range(img_canny.shape[1]):
            if img_canny[i][j] >= skipping_threshold:
                chuvi += 1
    list_c.append(chuvi)
plt.hist(list_c, 30, [0, 4000]); plt.show()
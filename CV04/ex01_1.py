import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def basic_sobel_edge_detection(image_path, blur_ksize=5, sobel_ksize=1, skipping_threshold=30):
    # Đọc ảnh đầu vào
    img = cv2.imread(image_path)
    # Chuyển ảnh thành ảnh đa mức xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Lọc ảnh đa mức xám bằng bộ lọc Gauss với kích thước phụ thuộc vào blur_ksize
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Thực hiện Sobel theo phương x
    img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    # Thực hiện Sobel theo phương y
    img_sobely = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # Tổng hợp ảnh Sobel từ phương x và y
    img_sobel = np.sqrt(img_sobelx**2 + img_sobely**2)

    # Cắt ngưỡng
    img_sobel = np.uint8(img_sobel)
    img_sobel[img_sobel < skipping_threshold] = 0
    img_sobel[img_sobel >= skipping_threshold] = 255

    return img_sobel

image_path = 'images/seed.png'
gray = cv2.imread(image_path, 0)
img_sobel = basic_sobel_edge_detection(image_path, 7, 1, 30)

plt.subplot(121),plt.imshow(gray, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_sobel,cmap = 'gray')
plt.title('Sobel Image'), plt.xticks([]), plt.yticks([])
plt.show()
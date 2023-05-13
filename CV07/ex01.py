import os
import numpy as np 
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

pathL = "sample/left/"
pathR = "sample/right/"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chuẩn bị object points giống như: (0,0,0), (1,0,0), ...., (10,6,0)
# Kích thước ô bàn cờ: 10x7
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(1,28)):
    imgL = cv2.imread(pathL+"img%d.png"%i)
    imgR = cv2.imread(pathR+"img%d.png"%i)
    imgL_gray = cv2.imread(pathL+"img%d.png"%i,0)
    imgR_gray = cv2.imread(pathR+"img%d.png"%i,0)

    outputL = imgL.copy()
    outputR = imgR.copy()
    
    # tìm các góc trên ảnh ô bàn cờ tương ứng cho camera trái và phải
    # gán các cặp điểm tương ứng giữa tọa độ thật object point (3D) và và tọa trên trên ảnh image point (2D) trên ảnh trái và ảnh phỉa
    # obj_pts, img_ptsL, img_ptsR
    ############### YOUR CODE HERE ########
    retR, cornersR = cv2.findChessboardCorners(imgR_gray, (9,6), None)
    retL, cornersL = cv2.findChessboardCorners(imgL_gray, (9,6), None)

    # cập nhật các tọa độ object và tọa độ ảnh: obj_pts, img_ptsL, img_ptsR
    if retR == True and retL == True:
        obj_pts.append(objp)
        
        cornersR = cv2.cornerSubPix(imgR_gray, cornersR, (11,11), (-1,-1), criteria)
        img_ptsR.append(cornersR)
        
        cornersL = cv2.cornerSubPix(imgL_gray, cornersL, (11,11), (-1,-1), criteria)
        img_ptsL.append(cornersL)
        
        outputL = cv2.drawChessboardCorners(outputL, (9,6), cornersL, retL)
        outputR = cv2.drawChessboardCorners(outputR, (9,6), cornersR, retR)
    ###########################################

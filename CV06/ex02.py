import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import math

images=glob.glob('chessboard/*.png')
column=2
row=int(math.ceil(len(images)/column))
column, row
plt.figure(figsize=(15,30))
for i,fname in enumerate(images):
    img=cv2.imread(fname)
    plt.subplot(row,column,i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(os.path.basename(fname))
plt.show()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01) # điều kiện dừng thuât toán sau số lẩn lặp tối đa hoặc mức độ hội tụ của t toán đạt được

# chessboard dimension (12 x 8) however we need objpoints and imgpoints to have same number of entries and of same size
# Kích thước của bàn cờ (12x8) tuy nhiên chúng ta cần objpoints và impoints có cùng số phần tử và cùng kích thước
cbrow = 11
cbcolumn = 7

# Chuẩn bị object points giống như: (0,0,0), (1,0,0), ...., (10,6,0). Tọa độ Z = 0 
objp = np.zeros((cbrow*cbcolumn, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbrow, 0:cbcolumn].T.reshape(-1, 2)

# Mảng đề lưu object points và image points từ toàn bộ các ảnh đảm bảo 2 mảng này có cùng kích thước
objpoints = []  # lưu trữ tọa độ 3d trong không gian thế giới thực (3d point in real world space)
imgpoints = []   # Lưu tọa độ 2D trên ảnh (2d point in image plane)

images = glob.glob('chessboard/*.png')
i = 0
for fname in images:
    i = i + 1
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tìm góc của bàn cờ (find the chessboard corners)
    ret, corners = cv2.findChessboardCorners(gray, (11, 7), flags=cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    
    # Nếu tìm thấy, thêm vào các điểm của đối tượng, điểm của ảnh (sau khi tinh chỉnh) (if found, add object points, image points (after refining them))
    if ret == True:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)
        
        # Vẽ và hiển thị các góc (Draw and display the corners)
        img = cv2.drawChessboardCorners(img, (cbrow, cbcolumn), corners2, ret)
        cv2.imshow('Image', img)
        #cv2.waitKey(0)
    else:
        print(f"Cannot find the chessboard corners for {fname}. Continue..")


cv2.destroyAllWindows()

# Các góc được nối với nhau theo thứ tự trong mang impoints, tương ứng với thứ tự các điểm 3D được lưu trong objpoint

plt.figure(figsize=(20,20))
plt.imshow(img)
plt.show()

# mtx: camera metrix (camera intrinsic matrix  )
# dist: distorsion coefficients 
# rvecs: Output vector of rotation vectors (Rodrigues ) estimated for each pattern view
# tvecs : Output vector of translation vectors estimated for each pattern view

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL) 
dist = np.array(dist)
##### UNDISTORTION #####
# UPDATE: Check if result folder exists. If no, make new one. 
output_folder = "chessboard/result"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    
img1 = cv2.imread('chessboard/scene00451.png')
h, w = img1.shape[:2]

# Tìm ma trận tham số trong của camera (không biến dạng) từ tham số camera cũ và hệ số biến dạng

# Compute the new camera intrinsic matrix based on the free scaling parameter (alpha)
# alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) 
# and 1 (when all the source image pixels are retained in the undistorted image)

alpha = 1
# compute the optimal new camera matrix based on the previously calculated distortion coefficients and camera matrix
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])
print("newcameramtx = \n", newcameramtx)

# The original camera intrinsic matrix, distortion coefficients, the computed new camera intrinsic matrix, 
# and newImageSize should be passed to initUndistortRectifyMap to produce the maps for remap.
resultImg = glob.glob('chessboard/*.png')
for rimg in resultImg:
    img = cv2.imread(rimg)
    
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)  # Computes the undistortion and rectification transformation map. 
            # The function computes the joint undistortion and rectification transformation 
            # and represents the result in the form of maps for remap. 
            # The undistorted image looks like original, as if it is captured with 
            # a camera using the camera matrix =newCameraMatrix and zero distortion.
            
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR) #It is the process of taking pixels from one place in the image and locating them in another position in a new image.
    
    # Ghi ảnh kết quả
    path = f"{output_folder}/Calibresult_" + \
        os.path.basename(rimg)
    cv2.imwrite(path, dst)

# Lỗi phép chiếu (re-projection error)
mean_error = 0
for i in range(len(objpoints)): # for each image, len(objpoints) = number of images used for calibration
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist) # chiếu 3D points lên image plan khi đã biết tham số trong và tham số ngoài 
    
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2) # tính sai số (L2) giữa các detected points (corners) và kết quả phép chiếu 
    mean_error += error
print("Mean error", mean_error/len(objpoints))

#cv2.waitKey(0)
#cv2.destroyAllWindows()
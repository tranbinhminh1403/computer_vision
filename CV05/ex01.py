import cv2
import numpy as np
import os
from PIL import Image

def detect_corner(image_path, blockSize=2, ksize=3, k=0.04, threshold=0.01):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,blockSize,ksize,k)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>threshold*dst.max()]=[0,0,255]
    
    # output file
    relative_path, image_basename = os.path.split(image_path)
    output_file = os.path.join(relative_path, 'corner_'+image_basename)
    
    cv2.imwrite(output_file,img)
    return output_file


# out_path = detect_corner('images/chessboard.jpg')
# out_path = detect_corner('images/sudoku/png')
out_path = detect_corner('images/house.jpg')
img = Image.open(out_path)
img.show()
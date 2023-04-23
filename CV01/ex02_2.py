import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.system('cls' if os.name == 'nt' else 'clear')
abs_path=[]
rel_path=[]

for file_name in os.listdir('image'):
    abs_file_path = os.path.abspath(os.path.join('image', file_name))
    abs_path.append(abs_file_path)
    rel_file_path = os.path.join('image', file_name)
    rel_path.append(rel_file_path)

img = cv2.imread(rel_path[2])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()
print(gray.shape)
histogram = np.zeros((256, ))
print(histogram.shape)
#ex2.2
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        histogram[gray[i,j]] += 1
        
plt.plot(histogram)
plt.show()
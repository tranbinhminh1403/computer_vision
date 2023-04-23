import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

#ex2
os.system('cls' if os.name == 'nt' else 'clear')
abs_path=[]
rel_path=[]

for file_name in os.listdir('image'):
    abs_file_path = os.path.abspath(os.path.join('image', file_name))
    abs_path.append(abs_file_path)
    rel_file_path = os.path.join('image', file_name)
    rel_path.append(rel_file_path)

# img = cv2.imread(rel_path[0])
# plt.imshow(img)
# plt.show()

# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(type(img2))
# print(img2.shape)
# plt.imshow(img2)
# plt.show()

# img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(type(img3))
# print(img3.shape)
# plt.imshow(img3, cmap='gray')
# plt.show()

# img3 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# # cv2.imshow('image', img3)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# plt.imshow(img3)
# plt.show()

# plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
# plt.show()

# img3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # cv2.imshow('image', img3)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# plt.imshow(img3)
# plt.show()

# img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
# plt.imshow(img3)
# plt.show()

# img4 = cv2.resize(img, (100, 100))
# print(img4.shape)
# plt.imshow(img4)
# plt.show()

img = cv2.imread(rel_path[0])

b_channel = img.copy()
# set green and red channels to 0
b_channel[:, :, 1] = 0
b_channel[:, :, 2] = 0
print(b_channel)

g_channel = img.copy()
# set blue and red channels to 0
g_channel[:, :, 0] = 0
g_channel[:, :, 2] = 0

r_channel = img.copy()
# set blue and green channels to 0
r_channel[:, :, 0] = 0
r_channel[:, :, 1] = 0

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(b_channel, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(g_channel, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(r_channel, cv2.COLOR_BGR2RGB))
plt.show()
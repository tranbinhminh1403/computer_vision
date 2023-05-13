import cv2
from PIL import Image

img = cv2.imread('images/ville01002.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp, des = sift.detectAndCompute(gray,None)

print('Number of keypoints:', len(kp))
print(kp[0].pt)
print(kp[0].size)
print(kp[0].angle)

print('Descriptor shape = ', des.shape)
print('Descriptor size = ', sift.descriptorSize())

img=cv2.drawKeypoints(gray,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints.jpg',img)
Image.open('sift_keypoints.jpg').show()

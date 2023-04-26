import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def get_gaussian_low_pass_filter(shape, cutoff):
    """Computes a gaussian low pass mask
    takes as input:
    shape: the shape of the mask to be generated
    cutoff: the cutoff frequency of the gaussian filter (sigma)
    returns a gaussian low pass mask"""

    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns), dtype=float)
    mid_R, mid_C = int(rows/2), int(columns/2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_R)**2 + (j - mid_C)**2)
            mask[i, j] = np.exp(-d**2/(2*(d0**2)))

    return mask


input_image = cv2.imread('images/messi.jpg', 0)

fft = np.fft.fft2(input_image)
shift_fft = np.fft.fftshift(fft)

rows, cols = input_image.shape[:2]


cutoff = 20
mask = get_gaussian_low_pass_filter((rows, cols), cutoff)


filtered_fft = shift_fft * mask


filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(input_image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title('Filtered Image')
plt.show()
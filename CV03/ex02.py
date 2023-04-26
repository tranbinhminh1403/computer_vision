import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# def get_ideal_high_pass_filter(shape, cutoff):
#     """Computes a Ideal high pass mask
#     takes as input:
#     shape: the shape of the mask to be generated
#     cutoff: the cutoff frequency of the ideal filter
#     returns a ideal high pass mask"""

#     d0 = cutoff
#     rows, columns = shape
#     mask = np.zeros((rows, columns), dtype=int)
#     mid_R, mid_C = int(rows/2), int(columns/2)
#     for i in range(rows):
#         for j in range(columns):
#             d = math.sqrt((i - mid_R)**2 + (j - mid_C)**2)
#             if d <= d0:
#                 mask[i, j] = 0
#             else:
#                 mask[i, j] = 1

#     return mask

# input_image = cv2.imread('images/messi.jpg', 0)

# fft = np.fft.fft2(input_image)
# shift_fft = np.fft.fftshift(fft)

# # showing purpose only
# mag_dft = np.log(np.abs(shift_fft))
# plt.imshow(mag_dft, cmap='gray')
# plt.show()

# rows, cols = input_image.shape
# cutoff = 50

# # Prepare low pass filter mask
# mask = get_ideal_high_pass_filter((rows, cols), cutoff)
# filtered_image = np.multiply(mask, shift_fft)

# # showing purpose only
# mag_filtered_dft = np.log(np.abs(filtered_image)+1)
# plt.imshow(mag_filtered_dft, cmap='gray')
# plt.show()

def get_ideal_high_pass_filter(shape, cutoff):
    """Computes a Ideal high pass mask
    takes as input:
    shape: the shape of the mask to be generated
    cutoff: the cutoff frequency of the ideal filter
    returns a ideal high pass mask"""

    d0 = cutoff
    rows, columns = shape
    mask = np.zeros((rows, columns), dtype=int)
    mid_R, mid_C = int(rows/2), int(columns/2)
    for i in range(rows):
        for j in range(columns):
            d = math.sqrt((i - mid_R)**2 + (j - mid_C)**2)
            if d <= d0:
                mask[i, j] = 0
            else:
                mask[i, j] = 1

    return mask


input_image = cv2.imread('images/messi.jpg', 0)

fft = np.fft.fft2(input_image)
shift_fft = np.fft.fftshift(fft)
# Testing the high pass filter
highpass_cutoff = 40
highpass_mask = get_ideal_high_pass_filter(input_image.shape, highpass_cutoff)

# Applying high pass filter to the image
highpass_fft = np.multiply(highpass_mask, shift_fft)
filtered_highpass_image = np.fft.ifft2(np.fft.ifftshift(highpass_fft)).real

# Displaying the original and filtered image side by side
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].imshow(input_image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(filtered_highpass_image, cmap='gray')
ax[1].set_title('High Pass Filtered Image')
plt.show()
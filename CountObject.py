import numpy as np
import cv2
import matplotlib.pyplot as plt

# TIEN XU LY

# imread(path,flag)
# flag 1  = anh mau`
# flag 0  = anh da muc xam
# image = cv2.imread('image/image1.jpeg', 1)
# image = cv2.imread('image/image2.jpeg', 1)
# image = cv2.imread('image/image3.jpeg', 1)
# image = cv2.imread('image/image4.jpeg', 1)
image = cv2.imread('image/18223995.jpg', 1)
# image = cv2.imread('image/Tolentino-Cats.jpg', 1)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_correct = np.array(255 * (grayImage / 255) ** 1.2 , dtype='uint8')

img_equal_hist = cv2.equalizeHist(grayImage)

# Local adaptative threshold use gray correct
# cv2.THRESH_BINARY: If pixel intensity is greater than the set threshold, value set to 255, else set to 0 (black).
# cv2.THRESH_BINARY_INV: Inverted or Opposite case of cv2.THRESH_BINARY.
# cv.THRESH_TRUNC: If pixel intensity value is greater than threshold, it is truncated to the threshold. The pixel values are set to be the same as the threshold. All other values remain the same.
# cv.THRESH_TOZERO: Pixel intensity is set to 0, for all the pixels intensity, less than the threshold value.
# cv.THRESH_TOZERO_INV: Inverted or Opposite case of cv2.THRESH_TOZERO.

thresh = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 10)
# thresh = cv2.adaptiveThreshold(img_equal_hist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 30)
# ret1, thresh = cv2.threshold(grayImage, threshold_value, 255, cv2.THRESH_BINARY_INV)
# ret, thresh = cv2.threshold(img_equal_hist,120,255,cv2.THRESH_BINARY)
# thresh = cv2.bitwise_not(thresh)

# Dilatation et erosion
kernel = np.ones((15, 15), np.uint8)

img_dilation = cv2.dilate(thresh, kernel, iterations=1)
img_erode = cv2.erode(img_dilation, kernel, iterations=1)

# clean all noise after dilatation and erosion
# image_after_erode_dilation = cv2.medianBlur(img_dilation, 7)
image_after_erode_dilation = cv2.medianBlur(img_erode, 7)


# Labeling
ret2, labels = cv2.connectedComponents(image_after_erode_dilation)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

f, axes = plt.subplots(2, 3, figsize=(25, 10))
axes[0][0].imshow(image)
axes[0][0].set_title('origin')
axes[0][1].imshow(grayImage, cmap="gray", vmin=0, vmax=255)
axes[0][1].set_title('Grayscale image')
axes[0][2].imshow(img_equal_hist, cmap="gray", vmin=0, vmax=255)
axes[0][2].set_title('Histogram qualization')
axes[1][0].imshow(thresh, cmap="gray", vmin=0, vmax=255)
axes[1][0].set_title('Threshold')
# axes[1][1].imshow(img_dilation, cmap="gray", vmin=0, vmax=255)
axes[1][1].imshow(image_after_erode_dilation, cmap="gray", vmin=0, vmax=255)
axes[1][1].set_title('Dilatation + erosion')
axes[1][2].imshow(labeled_img)
axes[1][2].set_title('Objects counted:' + str(ret2 - 1))
plt.show()

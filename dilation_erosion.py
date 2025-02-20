# -*- coding: utf-8 -*-
"""dilation-erosion.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AO3aV7dHBvgfdMAVUu5hhURXzjuuJ5JG
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
img = cv2.imread('/content/e.png', 0)
h, w = img.shape

# Erosion
k = 11
SE = np.ones((k, k), dtype=np.uint8)
c = (k - 1) // 2

img_erosion = np.zeros((h, w), dtype=np.uint8)

for i in range(c, h - c):
    for j in range(c, w - c):
        temp = img[i - c:i + c + 1, j - c:j + c + 1]
        l = temp * SE
        img_erosion[i][j] = np.min(l)

# Display the original image
cv2_imshow(img)

# Display the image after erosion
cv2_imshow(img_erosion)

# Dialation
kn = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
c = 1

img_dilation = np.zeros((h, w), dtype=np.uint8)

for i in range(c, h - c):
    for j in range(c, w - c):
        temp = img[i - c:i + c + 1, j - c:j + c + 1]
        l = temp * kn
        img_dilation[i][j] = np.max(l)

# Display the image after dilation
cv2_imshow(img_dilation)


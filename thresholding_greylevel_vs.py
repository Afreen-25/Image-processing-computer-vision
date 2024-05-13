import cv2
import numpy as np

img1 = cv2.imread('C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg')

grey_img1 = cv2.imread('C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg', 0)
h, w = grey_img1.shape

# Digital Negative
neg_img = 255 - grey_img1

# Thresholding
t = 50
threshold_img = np.zeros((h, w), np.uint8)
for i in range(h):
    for j in range(w):
        if grey_img1[i][j] > t:
            threshold_img[i][j] = 255
        else:
            threshold_img[i][j] = 0

# Grey level Slicing with BG
a = 150
b = 175
wbg_img = np.zeros((h, w), np.uint8)
for i in range(h):
    for j in range(w):
        r = grey_img1[i][j]
        if a <= r <= b:
            wbg_img[i][j] = 255
        else:
            wbg_img[i][j] = r

# Grey level Slicing without BG
wobg_img = np.zeros((h, w), np.uint8)
for i in range(h):
    for j in range(w):
        if a <= grey_img1[i][j] <= b:
            wobg_img[i][j] = 255
        else:
            wobg_img[i][j] = 0

cv2.imshow('Digital Negative', neg_img)
cv2.waitKey(0)

cv2.imshow('Threshold Image', threshold_img)
cv2.waitKey(0)

cv2.imshow('Grey Level Slicing with BG', wbg_img)
cv2.waitKey(0)

cv2.imshow('Grey Level Slicing without BG', wobg_img)
cv2.waitKey(0)

cv2.destroyAllWindows()

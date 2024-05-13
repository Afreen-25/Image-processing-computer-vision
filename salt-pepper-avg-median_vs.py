import cv2
import numpy as np
import math
import random

img = cv2.imread('C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg',0)
h,w = img.shape

# Averaging Filter

noise = np.random.normal(0,50,size = img.shape)
noisy_img = img + noise

cv2.imshow('Noisy Image',noisy_img)
cv2.waitKey()

mask = np.ones((3,3),dtype=int)
mask = mask/9

avg_filtered_img = np.zeros(img.shape,np.uint8)

for i in range(h-1):
    for j in range(w-1):
        temp = img[i-1][j-1]*mask[0][0] + img[i-1][j]*mask[0][1] + img[i-1][j+1]*mask[0][2] + img[i][j-1]*mask[1][0] + img[i][j]*mask[1][1] + img[i][j+1]*mask[1][2] + img[i+1][j-1]*mask[2][0] + img[i+1][j]*mask[2][1] + img[i+1][j+1]*mask[2][2]
        avg_filtered_img[i][j] = temp

print(mask)
print(img == avg_filtered_img)
cv2.imshow('Average filtered Image',avg_filtered_img)
cv2.waitKey()

# Median Filter

snp = img.copy()

for i in range(35000):
    x = random.randint(0,w-1)
    y = random.randint(0,h-1)
    snp[y][x] = 255

for i in range(35000):
    x = random.randint(0,w-1)
    y = random.randint(0,h-1)
    snp[y][x] = 0

cv2.imshow('Salt and pepper noisy image',snp)
cv2.waitKey()

snp_removed_img = np.zeros((h,w),np.uint8)
for i in range(1,h-1):
    for j in range(1,w-1):
        temp = [snp[i-1][j-1],
                snp[i-1][j],
                snp[i-1][j+1],
                snp[i][j-1],
                snp[i][j],
                snp[i][j+1],
                snp[i+1][j-1],
                snp[i+1][j],
                snp[i+1][j+1]]
        temp = sorted(temp)
        snp_removed_img[i][j] = temp[4]

cv2.imshow('Salt and pepper noise removed image',snp_removed_img)
cv2.waitKey()
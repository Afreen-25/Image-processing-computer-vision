import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\Admin\\Downloads\\e.png',0)
h,w = img.shape

# erossion
k = 11
SE = np.ones((k,k),dtype = np.uint8)
c = (k-1)//2

img_erossion = np.zeros((h,w),dtype=np.uint8)

for i in range(c,h-c):
    for j in range(c,w-c):
        temp = img[i-c:i+c+1,j-c:j+c+1]
        l = temp * SE
        img_erossion[i][j] = np.min(l)

plt.imshow(img,cmap='gray')
plt.show()

plt.imshow(img_erossion,cmap='gray')
plt.show()

# dialation

kn = np.array([[1,0,1],[1,1,1],[0,1,0]])
c = 1

img_dialation = np.zeros((h,w),dtype=np.uint8)

for i in range(c,h-c):
    for j in range(c,w-c):
        temp = img[i-c:i+c+1,j-c:j+c+1]
        l = temp * kn
        img_dialation[i][j] = np.max(l)

plt.imshow(img_dialation,cmap='gray')
plt.show()
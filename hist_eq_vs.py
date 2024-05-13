import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread('C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg',0)
h,w = img.shape

cv2.imshow('Original image', img)
cv2.waitKey()

# Histogram Equalization

def getcount(img):
    count = [0]*256
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            k = img[i][j]
            count[k] = count[k] + 1
    
    r = [*range(0,256,1)]
    return r,count

n,nk = getcount(img)

s = sum(nk)

pdf = []
for i in nk:
    pdf.append(i/s)

c = 0
cdf = []
for i in pdf:
    c = c + i
    cdf.append(c)

cdf255 = []
for i in cdf:
    cdf255.append(i*255)

op = []
for i in cdf255:
    op.append(math.ceil(i))

hist_equalized = np.zeros(img.shape,np.uint8)

for i in range(h):
    for j in range(w):
        k = img[i][j]
        hist_equalized[i][j] = op[k]

cv2.imshow('Histogram Equalized image',hist_equalized)
cv2.waitKey()

r,count = getcount(img)
plt.stem(r,count)
plt.show()

r,count = getcount(hist_equalized)
plt.stem(r,count)
plt.show()
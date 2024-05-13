import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg',0)
h,w = img.shape

cv2.imshow('Original image', img)
cv2.waitKey()

# histogram equalization
rmax = img.max()
rmin = img.min()
smax = 255
smin = 0
print("Before Histogram Stretching",rmax,rmin)

hist_streched = np.zeros((h,w),np.uint8)

for i in range(h):
    for j in range(w):
        r = img[i][j]
        hist_streched[i][j] = (r-rmin+smin)*((smax-smin)/(rmax-rmin))

cv2.imshow('Histogram Streched image', hist_streched)
cv2.waitKey()

print("After Histogram Stretching",hist_streched.max(),hist_streched.min())

# plotting histogram

def getcount(img):
    count = [0]*256
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            k = img[i][j]
            count[k] = count[k] + 1
    
    r = [*range(0,256,1)]
    return r,count

r,count = getcount(img)

plt.stem(r,count)
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel count')
plt.title('Histogram of original image')
plt.show()

r,count = getcount(hist_streched)

plt.stem(r,count)
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel count')
plt.title('Histogram of stretched image')
plt.show()
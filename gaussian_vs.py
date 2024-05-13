import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg',0)

h,w = img.shape

F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)

H = np.zeros((h,w),dtype = np.float32)
D0 = 50

for i in range(h):
    for j in range(w):
        D = np.sqrt((i-h/2)**2 + (j-w/2)**2)
        H[i][j] = np.exp(-(D**2)/(2*(D0**2)))

# Gaussian high pass
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

plt.imshow(H,cmap='gray')
plt.show()

plt.imshow(g,cmap='gray')
plt.show()

# Gaussian low pass
H = 1-H
Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

plt.imshow(H,cmap='gray')
plt.show()

plt.imshow(g,cmap='gray')
plt.show()
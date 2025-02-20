# -*- coding: utf-8 -*-
"""arith_log.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19xdpv3P5RqHQnIRsyGEwqFBondAl34sD
"""



from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import cv2

(x_train, _), (_, _) = cifar10.load_data()

image1 = x_train[np.random.randint(0, len(x_train))]
image2 = x_train[np.random.randint(0, len(x_train))]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title('Original Image - 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.title('Original Image - 2')
plt.axis('off')

plt.show()

"""Convert to Grayscale"""

def rgb_to_grayscale(image):
  height,width,channels = image.shape
  grayscaleImage = np.zeros((height,width))
  for x in range(height):
    for y in range(width):
        pixelValue = sum(image[x,y])/channels
        grayscaleImage[x,y] = pixelValue
  return grayscaleImage

grayImage1 = rgb_to_grayscale(image1)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(grayImage1,cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.show()

"""Crop an Image"""

def crop_image(image, x, y):
    height,width,_ = image.shape
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image

croppedImage = crop_image(image1,20,10)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(croppedImage)
plt.title('Cropped Image')

plt.show()

"""Arithmetic Operations"""

def addImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    addedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                addedImage[y, x, c] = min(int(image1[y, x, c]) + int(image2[y, x, c]), 255)
    return addedImage

def subtractImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    subtractedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                subtractedImage[y, x, c] = max(int(image1[y, x, c]) - int(image2[y, x, c]), 0)
    return subtractedImage

def multiplyImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    multipliedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                multipliedImage[y, x, c] = min(int(image1[y, x, c]) * int(image2[y, x, c]), 255)
    return multipliedImage

def divideImages(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same size"
    height, width, channels = image1.shape
    dividedImage = image1.copy()
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                if int(image2[y, x, c]) != 0:
                    dividedImage[y, x, c] = min(int(image1[y, x, c]) / int(image2[y, x, c]), 255)
                else:
                    dividedImage[y, x, c] = 255  # Handle division by zero
    return dividedImage

addedImage = addImages(image1,image2)
subtractedImage = subtractImages(image1,image2)
multipliedImage = multiplyImages(image1,image2)
dividedImage = divideImages(image1,image2)

plt.figure(figsize=(6, 6))

plt.subplot(4, 2, 1)
plt.imshow(addedImage)
plt.title('Added Image Scratch')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.imshow(cv2.add(image1, image2))
plt.title('Added Image CV2')
plt.axis('off')

plt.subplot(4, 2, 3)
plt.imshow(subtractedImage)
plt.title('Subtracted Image Scratch')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(cv2.subtract(image1, image2))
plt.title('Subtracted Image CV2')
plt.axis('off')

plt.subplot(4, 2, 5)
plt.imshow(multipliedImage)
plt.title('Multiplied Image Scratch')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.imshow(cv2.multiply(image1, image2))
plt.title('Multiplied Image CV2')
plt.axis('off')

plt.subplot(4, 2, 7)
plt.imshow(dividedImage)
plt.title('Divided Image Scratch')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.imshow(cv2.divide(image1, image2))
plt.title('Divided Image CV2')
plt.axis('off')

plt.tight_layout()
plt.show()
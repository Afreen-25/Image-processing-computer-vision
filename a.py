import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(_, _), (test_images, _) = cifar10.load_data()

# High pass filter mask
mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Function to apply high pass filter
def apply_high_pass_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    hp_img = np.zeros((h, w), np.float32)
    
    # Apply the high pass filter
    for i in range(1, h-1):
        for j in range(1, w-1):
            temp = (img_gray[i-1, j-1] * mask[0, 0] + img_gray[i-1, j] * mask[0, 1] + img_gray[i-1, j+1] * mask[0, 2] +
                    img_gray[i, j-1] * mask[1, 0] + img_gray[i, j] * mask[1, 1] + img_gray[i, j+1] * mask[1, 2] +
                    img_gray[i+1, j-1] * mask[2, 0] + img_gray[i+1, j] * mask[2, 1] + img_gray[i+1, j+1] * mask[2, 2])
            hp_img[i, j] = temp
            
    # Normalize the filtered image to 0-255
    hp_img = cv2.normalize(hp_img, None, 0, 255, cv2.NORM_MINMAX)
    return hp_img.astype(np.uint8)

# Display the original and high-pass filtered images
plt.figure(figsize=(20, 2))
for i in range(10):
    # Apply high pass filter
    filtered_img = apply_high_pass_filter(test_images[i])
    
    
    # Display high-pass filtered image
    plt.subplot(2, 10, i+11)
    plt.imshow(filtered_img, cmap='gray')
    plt.axis('off')

plt.show()

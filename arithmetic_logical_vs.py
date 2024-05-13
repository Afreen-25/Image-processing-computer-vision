import cv2
import numpy as np

# Load the image using OpenCV
image_path = 'C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg'
image = cv2.imread(image_path)

# Convert the image to a NumPy array
image_array = np.array(image, dtype=np.uint8)

# Display the loaded image
cv2.imshow('Original Image', image_array)
cv2.waitKey(0)

# Convert the image to grayscale
grayscale_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
cv2.imshow('Grayscale Image', grayscale_image.astype(np.uint8))
cv2.waitKey(0)

# Implement image cropping
crop_x, crop_y, crop_width, crop_height = 50, 50, 200, 200
cropped_image = image_array[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)

# Perform arithmetic operations
addition_image = np.clip(image_array.astype(np.int16) + 50, 0, 255).astype(np.uint8)
cv2.imshow('Addition Image', addition_image)
cv2.waitKey(0)

subtraction_image = np.clip(image_array.astype(np.int16) - 50, 0, 255).astype(np.uint8)
cv2.imshow('Subtraction Image', subtraction_image)
cv2.waitKey(0)

multiplication_image = np.clip(image_array.astype(np.int16) * 1.5, 0, 255).astype(np.uint8)
cv2.imshow('Multiplication Image', multiplication_image)
cv2.waitKey(0)

division_image = np.clip(image_array.astype(np.int16) // 2, 0, 255).astype(np.uint8)
cv2.imshow('Division Image', division_image)
cv2.waitKey(0)

# Perform logical operations
thresh = 128
and_image = np.where(image_array > thresh, 255, 0).astype(np.uint8)
cv2.imshow('AND Image', and_image)
cv2.waitKey(0)

or_image = np.where(image_array > thresh, image_array, thresh).astype(np.uint8)
cv2.imshow('OR Image', or_image)
cv2.waitKey(0)

not_image = np.invert(image_array)
cv2.imshow('NOT Image', not_image)
cv2.waitKey(0)

xor_image = np.where(image_array > thresh, 0, 255).astype(np.uint8) ^ image_array
cv2.imshow('XOR Image', xor_image)
cv2.waitKey(0)

# Perform bitshift operations
left_shift_image = np.left_shift(image_array, 1)
cv2.imshow('Left Shift Image', left_shift_image)
cv2.waitKey(0)

right_shift_image = np.right_shift(image_array, 1)
cv2.imshow('Right Shift Image', right_shift_image)
cv2.waitKey(0)

cv2.destroyAllWindows()  # Close all OpenCV windows

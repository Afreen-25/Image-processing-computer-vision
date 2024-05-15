import cv2
import matplotlib.pyplot as plt
image = cv2.imread("C:\\Users\\Admin\\Downloads\\beautiful-nature-scenery-free-photo (1).jpg", 0) #grayscale
cv2.imshow("orig", image)
cv2.waitKey(0)


# Define the crop rectangle (x, y, w, h)
x, y, w, h = 100, 50, 300, 200

# Crop the image
cropped_image = image[y:y+h, x:x+w]

# Display the original and cropped images
cv2.imshow('Original Image', image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
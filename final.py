import numpy as np
import matplotlib.image as mpimg
import cv2
import pandas as pd
# test_image = cv2.imread('area_task.jpg')
test_image = cv2.imread('test.jpg')
image_grayscal = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)

image_hsv = cv2.cvtColor(test_image,cv2.COLOR_RGB2HSV)
h , s, v = cv2.split(image_hsv)



gray_blur = cv2.GaussianBlur(s, (15,15), 0)

binary_image = np.copy(gray_blur)
binary_image = cv2.Canny(gray_blur,0,90)

kernel = np.ones((3, 3), np.uint8)
binary_image = cv2.dilate(binary_image, kernel, iterations=5)
binary_image = cv2.erode(binary_image, kernel, iterations=1)

contours, reterval = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final_picture= np.zeros_like(test_image)
largest_contour = max(contours, key=cv2.contourArea)
black = np.zeros_like(test_image)

cnt=largest_contour[0]

x, y, w, h = cv2.boundingRect(largest_contour)
# cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


src_interest_pts = np.float32([[0, 0], [0, test_image.shape[1]], [test_image.shape[0], 0], [test_image.shape[0], test_image.shape[1]]])
Projective_interest_pts = np.float32([[x, y], [x, y+h], [x+w, y], [x+w, y+h]])

# Compute perspective transformation matrix
M = cv2.getPerspectiveTransform(Projective_interest_pts,src_interest_pts)

# Apply perspective transformation
rows, cols, _ = test_image.shape
Projectivedst = cv2.warpPerspective(test_image, M, (rows, cols))



di=cv2.resize(test_image,(700,700))
di=cv2.resize(Projectivedst,(700,700))
cv2.imshow('test',di)
cv2.waitKey(0)
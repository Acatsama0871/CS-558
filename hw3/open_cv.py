import cv2
import numpy as np

# Load the images
image1 = cv2.imread("/home/haohang/CS-558/hw3/data/uttower_left.jpg")
image2 = cv2.imread("/home/haohang/CS-558/hw3/data/uttower_right.jpg")
print(image1.shape)

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches = bf.match(descriptors1, descriptors2)

# Extract the matched keypoints
src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate the homography matrix using RANSAC
# homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
homography, status = cv2.estimateAffinePartial2D(src_points, dst_points)
print(homography)

result = cv2.warpAffine(image1, homography, (image2.shape[1], image2.shape[0]))
alpha = 0.5  # blending factor
blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

cv2.imwrite("/home/haohang/CS-558/hw3/result.jpg", blended_image)

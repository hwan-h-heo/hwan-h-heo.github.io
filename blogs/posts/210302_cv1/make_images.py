# Chapter 2: Image Filtering
# Topic: Linear and Non-Linear Filtering for Denoising

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Create a sample image and add salt-and-pepper noise
# Using a simple gray image with a black square
image = np.full((300, 300, 3), 128, dtype=np.uint8)
cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 0), -1)

# Add salt-and-pepper noise
salt_pepper = np.random.rand(image.shape[0], image.shape[1])
image[salt_pepper > 0.95] = 255  # Salt
image[salt_pepper < 0.05] = 0    # Pepper

# 2. Apply different filters
# Box filter (Averaging)
box_filtered = cv2.blur(image, (5, 5))

# Gaussian filter
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

# Median filter
median_filtered = cv2.medianBlur(image, 5)

# 3. Visualize results
titles = ['Original with Noise', 'Box Filter', 'Gaussian Filter', 'Median Filter']
images = [image, box_filtered, gaussian_filtered, median_filtered]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.suptitle("Comparison of Denoising Filters", fontsize=16)
# plt.show()
plt.savefig('./assets/image-1.jpg', dpi=200)
# The median filter is most effective at removing salt-and-pepper noise while preserving edges,
# whereas the box and Gaussian filters blur the noise along with the image features.

image = np.zeros((300, 500), dtype=np.uint8)
cv2.putText(image, "OpenCV", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (255), 10)
# Add "pepper" noise (holes)
image[100:150, 150:200] = 0
# Add "salt" noise (small specks)
cv2.circle(image, (400, 100), 5, 255, -1)
cv2.circle(image, (50, 50), 5, 255, -1)

# 2. Define a structuring element and apply operations
kernel = np.ones((7, 7), np.uint8)

# Erosion shrinks bright regions
erosion = cv2.erode(image, kernel, iterations=1)
# Dilation expands bright regions
dilation = cv2.dilate(image, kernel, iterations=1)
# Opening = erosion followed by dilation (removes salt noise)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# Closing = dilation followed by erosion (fills pepper holes)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 3. Visualize results
titles = ['Original Noisy', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [image, erosion, dilation, opening, closing]

plt.figure(figsize=(15, 7))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.suptitle("Morphological Operations", fontsize=16)
plt.savefig('./assets/image-2.jpg', dpi=200)

# 1. Load an image (or create one)
# Create a sample image with geometric shapes
image = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(image, (100, 100), 50, 255, 3)
cv2.rectangle(image, (150, 150), (250, 250), 255, -1)
cv2.line(image, (50, 250), (250, 50), 255, 5)

# 2. Apply Sobel and Canny detectors
# Sobel operator to find gradients
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
# Compute magnitude and normalize
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

# Canny edge detector
canny_edges = cv2.Canny(image, threshold1=100, threshold2=200)

# 3. Visualize results
titles = ['Original Image', 'Sobel Gradient Magnitude', 'Canny Edges']
images = [image, sobel_magnitude, canny_edges]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.suptitle("Edge Detection Techniques", fontsize=16)
plt.savefig('./assets/image-3.jpg', dpi=200)

image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.putText(image, 'CV', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)

# 2. Generate the Gaussian pyramid
pyramid_levels = 4
pyramid = [image]
temp_image = image
for i in range(pyramid_levels - 1):
    # Blur and downsample
    temp_image = cv2.pyrDown(temp_image)
    pyramid.append(temp_image)

# 3. Visualize the pyramid
plt.figure(figsize=(10, 5))
# Display images in a row
total_width = sum(img.shape[1] for img in pyramid)
display_image = np.ones((image.shape[0], total_width, 3), dtype=np.uint8) * 255

current_x = 0
for level in pyramid:
    h, w, _ = level.shape
    display_image[0:h, current_x:current_x+w] = level
    current_x += w

plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
plt.title(f'Gaussian Pyramid ({pyramid_levels} Levels)')
plt.xticks([]), plt.yticks([])
plt.savefig('./assets/image-4.jpg', dpi=200)

image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (250, 250), 255, 3)
cv2.rectangle(image, (100, 100), (200, 200), 255, 3)
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # For drawing colored circles

# 2. Apply Harris Corner Detector
gray = np.float32(image)
# harris_corners = cv2.cornerHarris(src, blockSize, ksize, k)
harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate the corner image to make the corners more visible
harris_corners = cv2.dilate(harris_corners, None)

# 3. Threshold and visualize the corners
# Mark the detected corners on the original image
image_color[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255] # Red

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection Results')
plt.xticks([]), plt.yticks([])
plt.savefig('./assets/image-5.jpg', dpi=200)

# 1. Load or create an image
image = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.putText(image, 'CV', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)
rows, cols, _ = image.shape

# 2. Define source and destination points for the transformation
# We will map a triangle to a different triangle
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Get the affine transformation matrix
M = cv2.getAffineTransform(pts1, pts2)

# Apply the transformation
warped_image = cv2.warpAffine(image, M, (cols, rows))

# 3. Visualize results
# Draw the triangles on the images for clarity
cv2.polylines(image, [np.int32(pts1)], True, (0, 255, 0), 3)
cv2.polylines(warped_image, [np.int32(pts2)], True, (0, 255, 0), 3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title('Affine Warped Image')
plt.xticks([]), plt.yticks([])

plt.suptitle("Affine Transformation", fontsize=16)
plt.savefig('./assets/image-6.jpg', dpi=200)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Create two views of the same scene
# Scene
scene = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(scene, (100, 100), (300, 300), (255, 0, 0), 5)
cv2.putText(scene, 'SRC', (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

# Create a second, warped view (e.g., from a different perspective)
M = np.float32([[1, -0.2, 50], [0.1, 0.9, 30]])
warped_scene = cv2.warpAffine(scene, M, (scene.shape[1], scene.shape[0]))

# 2. Find correspondences
# Initialize ORB detector
orb = cv2.ORB_create()
# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(scene, None)
kp2, des2 = orb.detectAndCompute(warped_scene, None)

# Use BFMatcher to find the best matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Keep only the best 50 matches for clarity
good_matches = matches[:50]

# 3. Find Homography and visualize
# Extract locations of good matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Find homography matrix using RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Draw matches
match_img = cv2.drawMatches(scene, kp1, warped_scene, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 7))
plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
plt.title('Feature Matching for Homography Estimation')
plt.savefig('./assets/image-7.jpg', dpi=200)

f = 800 # Focal length
cx, cy = 320, 240 # Principal point
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

# Camera Extrinsic Parameters [R|t] (camera looks along Z-axis)
R = np.identity(3) # No rotation
t = np.array([[0], [0], [5]]) # Camera is 5 units away from origin

# Define a 3D cube (8 vertices)
cube_points_3d = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
]).T # Transpose to get 3xN

# 2. Project the 3D points to the 2D image plane
# First, transform from world to camera coordinates
points_cam_coord = R @ cube_points_3d + t

# Then, project to 2D using the intrinsic matrix K
# The projection equation is: x_img = K * X_cam
projected_points_homogeneous = K @ points_cam_coord

# Convert from homogeneous to Cartesian coordinates by dividing by the last component (Z)
projected_points_2d = projected_points_homogeneous[:2, :] / projected_points_homogeneous[2, :]

# 3. Visualize the 2D projection
plt.figure(figsize=(8, 6))
# Plot projected points
plt.scatter(projected_points_2d[0, :], projected_points_2d[1, :], c='red', s=100)

# Draw lines to form the cube
edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
for edge in edges:
    p1 = projected_points_2d[:, edge[0]]
    p2 = projected_points_2d[:, edge[1]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')

plt.title('2D Projection of a 3D Cube')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.gca().invert_yaxis() # Image coordinates usually have y-axis pointing down
plt.grid(True)
plt.axis('equal')
plt.savefig('./assets/image-8.jpg', dpi=200)

try:
    imgL = cv2.imread('tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('tsukuba_r.png', cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None: raise FileNotFoundError
except FileNotFoundError:
    print("Tsukuba images not found. Creating a synthetic stereo pair.")
    # Create a simple scene with objects at different depths
    imgL = np.full((300, 400), 200, dtype=np.uint8)
    imgR = imgL.copy()
    # Near object (large disparity)
    cv2.rectangle(imgL, (100, 100), (150, 150), 100, -1)
    cv2.rectangle(imgR, (100 - 20, 100), (150 - 20, 150), 100, -1)
    # Far object (small disparity)
    cv2.circle(imgL, (250, 150), 30, 50, -1)
    cv2.circle(imgR, (250 - 5, 150), 30, 50, -1)

# 1. Initialize Stereo Block Matcher
# numDisparities must be divisible by 16
stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=15)

# 2. Compute disparity map
disparity = stereo.compute(imgL, imgR)

# 3. Visualize results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(imgL, 'gray')
plt.title('Left Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(imgR, 'gray')
plt.title('Right Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3)
# Normalize the disparity map for visualization
disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.imshow(disparity_visual, 'gray')
plt.title('Disparity Map')
plt.xticks([]), plt.yticks([])

# plt.suptitle("Stereo Vision - Disparity Calculation", fontsize=16)
plt.savefig('./assets/image-9.jpg', dpi=200)
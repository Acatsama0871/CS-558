import numpy as np
import cv2

def random_line_from_points(points):
    """ Select two random points and return the line equation coefficients a, b, c for ax + by + c = 0 """
    p1, p2 = points[np.random.choice(points.shape[0], 2, replace=False), :]
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    return a, b, c

def distance_from_line(point, line):
    """ Calculate the distance of a point from a line """
    a, b, c = line
    x, y = point
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

def ransac_line_fitting(points, iterations=1000, threshold=2):
    best_inlier_count = 0
    best_line = None

    for _ in range(iterations):
        line = random_line_from_points(points)
        inlier_count = sum(distance_from_line(p, line) < threshold for p in points)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line = line

    return best_line

# Example usage
points = np.random.rand(100, 2) * 100  # Generating some random points
best_line = ransac_line_fitting(points)
print("Best line coefficients (a, b, c):", best_line)

# If you want to visualize the result on an image
img = np.zeros((100, 100, 3), dtype=np.uint8)
for point in points:
    cv2.circle(img, tuple(point.astype(int)), 1, (255, 255, 255), -1)

if best_line:
    a, b, c = best_line
    # Drawing the line
    x0, y0 = 0, int(-c / b)
    x1, y1 = 99, int(-(a * 99 + c) / b)
    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

cv2.imshow("RANSAC Line Fitting", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
from rich import print


def create_test_image_with_points():
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (50, 50), 1, 255, -1)  # Point at (50, 50)
    cv2.circle(image, (150, 50), 1, 255, -1)  # Point at (150, 50)
    return image


def hough_line_detection(
    keypoint_image, theta_resolution=1.0, rho_resolution=1.0, min_dist=20
):
    height, width = keypoint_image.shape
    diag_len = np.ceil(np.sqrt(height**2 + width**2)).astype(
        int
    )  # Maximum possible rho value
    thetas = np.deg2rad(np.arange(-90, 90, step=theta_resolution))
    rhos = np.arange(-diag_len, diag_len, step=rho_resolution)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.int64)

    # Find edge points
    y_idxs, x_idxs = np.nonzero(keypoint_image)

    # Vote in the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(thetas)):
            rho = int(round(x * np.cos(thetas[j]) + y * np.sin(thetas[j]))) + diag_len
            accumulator[rho, j] += 1

    # Extract lines
    threshold = np.max(accumulator) * 0.5
    lines = []
    for rho, theta in zip(*np.where(accumulator > threshold)):
        actual_rho = (rho - diag_len) * rho_resolution
        if not any(abs(actual_rho - r) < min_dist for r, t in lines):
            lines.append((actual_rho, thetas[theta]))

    return lines


if __name__ == "__main__":
    test_image = create_test_image_with_points()
    cv2.imwrite("test_image.png", test_image)

    detected_lines = hough_line_detection(test_image)

    for rho, theta in detected_lines:
        print(
            f"Detected line: rho = {rho:.2f}, theta = {np.rad2deg(theta) % 360:.2f} degrees"
        )

    print("Expected line: rho = 50, theta = 90 degrees")

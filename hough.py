import cv2
import numpy as np
from rich import print
from typing import Tuple


def create_test_image_with_points():
    image = np.zeros((200, 200), dtype=np.uint8)
    # Two points on the horizontal line at y = 50
    cv2.circle(image, (50, 50), 1, 255, -1)  # type: ignore Point at (50, 50)
    cv2.circle(image, (150, 50), 1, 255, -1)  # type: ignore  Point at (150, 50)
    return image


def hough_transform_iter(
    keypoints: Tuple[np.ndarray, np.ndarray],
    img_height: int,
    img_width: int,
    theta_step: float = 1.0,
    rho_step: float = 1.0,
    min_dist: float = 2.0,
):
    # create the accumulator
    diag_dist = np.ceil(np.sqrt(img_height**2 + img_width**2)).astype(int)
    thetas = np.deg2rad(np.arange(-90, 90, step=theta_step))
    rhos = np.arange(-diag_dist, diag_dist, step=rho_step)
    accumulator = np.zeros((2 * diag_dist, len(thetas)), dtype=np.int64)

    # vote
    y, x = keypoints
    for i in range(len(x)):
        for j in range(len(thetas)):
            rho = (
                int(round(x[i] * np.cos(thetas[j]) + y[i] * np.sin(thetas[j])))
                + diag_dist
            )
            accumulator[rho, j] += 1

    # find the most salient line
    rho_index, theta_index = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    rho = rhos[rho_index]
    theta = thetas[theta_index]
    if rho < 0:
        rho = -rho
        theta = theta + np.pi

    # find the point with in the min_dist
    all_points = np.hstack((y.reshape(-1, 1), x.reshape(-1, 1)))
    dists = np.abs(
        all_points[:, 0] * np.sin(theta) + all_points[:, 1] * np.cos(theta) - rho
    )
    to_remove_index = np.where(dists < min_dist)[0]
    new_y = np.delete(y, to_remove_index)
    new_x = np.delete(x, to_remove_index)
    new_keypoints = (new_y, new_x)
    inliers = (y[to_remove_index], x[to_remove_index])

    return (rho, theta), new_keypoints, inliers


def hough_transform_line_detection(
    keypoint_image: np.ndarray,
    theta_step: float = 1.0,
    rho_step: float = 1.0,
    min_dist: float = 10.0,
    num_lines: int = 4,
):
    keypoints = np.nonzero(keypoint_image)

    lines = []
    inliers = []

    for _ in range(num_lines):
        cur_line, keypoints, cur_inlier = hough_transform_iter(
            keypoints=keypoints,  # type: ignore
            img_height=keypoint_image.shape[0],
            img_width=keypoint_image.shape[1],
            theta_step=theta_step,
            rho_step=rho_step,
            min_dist=min_dist,
        )
        lines.append(cur_line)
        inliers.append(cur_inlier)

        if len(keypoints[0]) == 0:
            print("No more keypoints to find")
            break
    return lines, inliers


if __name__ == "__main__":
    test_image = create_test_image_with_points()
    cv2.imwrite("test_image.png", test_image)

    detected_lines, inliers = hough_transform_line_detection(test_image)
    print(detected_lines)

    print("Expected line: rho = 50, theta = 90 degrees")
    for l in detected_lines:
        print(
            f"Detected line: rho = {l[0]:.2f}, theta = {np.rad2deg(l[1]) % 360:.2f} degrees"
        )

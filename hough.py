import cv2
import numpy as np
from rich import print
from typing import Tuple
from rich.console import Console

console = Console()


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

    return (rho, theta), new_keypoints, accumulator, inliers


def hough_transform_line_detection(
    keypoint_image: np.ndarray,
    theta_step: float = 1.0,  # very sensitive to this parameter
    rho_step: float = 1.0,  # very sensitive to this parameter
    min_dist: float = 10.0,
    num_lines: int = 4,
):
    keypoints = np.nonzero(keypoint_image)

    lines = []
    inliers = []
    accumulators = []

    for _ in range(num_lines):
        cur_line, keypoints, cur_accumulator, cur_inliers = hough_transform_iter(
            keypoints=keypoints,  # type: ignore
            img_height=keypoint_image.shape[0],
            img_width=keypoint_image.shape[1],
            theta_step=theta_step,
            rho_step=rho_step,
            min_dist=min_dist,
        )
        lines.append(cur_line)
        accumulators.append(cur_accumulator)
        inliers.append(cur_inliers)

        if len(keypoints[0]) == 0:
            console.print("[red] No more keypoints to find [/red]")
            break
    return lines, accumulators, inliers


if __name__ == "__main__":
    # test_image = create_test_image_with_points()
    # cv2.imwrite("test_image.png", test_image)
    image = cv2.imread("/home/haohang/CS-558/hw2/result/keypoint.png", 0)

    detected_lines, accumulators, inliers = hough_transform_line_detection(image)

    print("Expected line: rho = 50, theta = 90 degrees")
    for l in detected_lines:
        print(
            f"Detected line: rho = {l[0]:.2f}, theta = {np.rad2deg(l[1]) % 360:.2f} degrees"
        )

    # plot
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for cur_index, cur_result in enumerate(zip(detected_lines, accumulators, inliers)):
        cur_line, cur_accumulator, cur_inlier = cur_result
        y, x = cur_inlier
        # plot line
        a, b, c = np.cos(cur_line[1]), np.sin(cur_line[1]), -cur_line[0]
        x0, y0 = int(x.min()), int((-a * x.min() - c) / b)
        x1, y1 = int(x.max()), int((-a * x.max() - c) / b)
        cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), thickness=1)
        # plot accumulator
        scaled_accumulator = np.uint8(255 * cur_accumulator / np.max(cur_accumulator))
        cv2.imwrite(
            f"/home/haohang/CS-558/hw2/result/accumulator_line_{cur_index}.png",
            scaled_accumulator,  # type: ignore
        )  # type: ignore
    cv2.imwrite("/home/haohang/CS-558/hw2/result/detected_line.png", image)

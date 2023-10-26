import os
import cv2
import typer
import warnings
import itertools
import numpy as np
from typing import Union, Tuple, List
from PIL import Image
from numba import jit
from tqdm.auto import tqdm
from rich.console import Console


app = typer.Typer()
np.random.seed(42)
warnings.filterwarnings("ignore")
console = Console()


# helper function
def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img_gray = img.convert("L")
    return np.array(img_gray).astype(np.float64)


def save_image(path: str, image: np.ndarray):
    img = Image.fromarray(image)
    img.convert("L").save(path)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    i_h = image.shape[0]
    i_w = image.shape[1]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    # the kernel must have odd dimensions for simplicity
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("Kernel must have odd dimensions")
    if k_h > i_h or k_w > i_w:
        raise ValueError("Kernel cannot be larger than image")

    half_k_h = k_h // 2
    half_k_w = k_w // 2
    image = np.pad(image, (half_k_h, half_k_w), mode="edge")

    # convolution operation
    output = image.copy()
    for x, y in itertools.product(
        range(half_k_h, image.shape[0] - half_k_h),
        range(half_k_w, image.shape[1] - half_k_w),
    ):
        if ((x + half_k_h) > image.shape[0]) or ((y + half_k_w) > image.shape[1]):
            continue
        output[x, y] = (
            kernel
            * image[
                (x - half_k_h) : (x + half_k_h + 1), (y - half_k_w) : (y + half_k_w + 1)
            ]
        ).sum()

    return output[half_k_h:-half_k_h, half_k_w:-half_k_w]


# gaussian filter
@jit(nopython=True)
def gaussian_func(x: int, y: int, sigma: float, size: int) -> float:
    return (
        1
        / (2 * np.pi * sigma**2)
        * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2))
    )


def gaussian_filter(
    image: np.ndarray, sigma: float, size: Union[int, None] = None
) -> Tuple[np.ndarray, int]:
    # set size
    if size is None:
        size = int(np.floor(6 * sigma + 1))
    # get gaussian kernel
    kernel = np.zeros((size, size))
    for i, j in itertools.product(range(size), range(size)):
        kernel[i, j] = gaussian_func(x=i, y=j, sigma=sigma, size=size)
    kernel = kernel / kernel.sum()
    # convolve
    return convolve2d(image, kernel), size


# sobel operator
def sobel_operator(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # convolve
    return convolve2d(image, sobel_x), convolve2d(image, sobel_y)


# corner keypoint detector
def corner_keypoint_detector(
    image: np.ndarray,
    sigma: float = 6.0,
    gaussian_filter_size: Union[int, None] = None,
    threshold: Union[float, None] = None,
    threshold_quantile: Union[float, None] = 0.95,
):
    # param check
    if threshold is None:
        if threshold_quantile is None:
            raise ValueError("Either threshold or threshold quantile must be specified")
    elif threshold_quantile is not None:
        raise ValueError("Only one of threshold or threshold quantile can be specified")
    # gaussian smoothing
    result, _ = gaussian_filter(image=image, sigma=sigma, size=gaussian_filter_size)
    # second order derivatives
    i_x, i_y = sobel_operator(result)
    i_xx, i_xy = sobel_operator(i_x)
    i_yx, i_yy = sobel_operator(i_y)
    # hessian determinant
    det_hessian = i_xx * i_yy - i_xy * i_yx
    # normalize to [0, 255]
    det_hessian = np.floor(det_hessian / np.max(det_hessian) * 255)
    # thresholding
    if threshold is None:
        threshold = np.quantile(det_hessian, threshold_quantile)  # type: ignore
    det_hessian[det_hessian < threshold] = 0
    det_hessian[det_hessian >= threshold] = 1
    # non-maximum suppression
    for x in range(1, det_hessian.shape[0] - 1):
        for y in range(1, det_hessian.shape[1] - 1):
            if det_hessian[x, y] != np.max(
                det_hessian[x - 1 : x + 2, y - 1 : y + 2]
            ):  # type: ignore
                det_hessian[x, y] = 0
            elif det_hessian[x, y] != 0:  # if all zeros, keep it
                det_hessian[x - 1 : x + 2, y - 1 : y + 2] = 0  # type: ignore
                det_hessian[x, y] = 1
    # ignore border
    det_hessian[0, :] = 0
    det_hessian[-1, :] = 0
    det_hessian[:, 0] = 0
    det_hessian[:, -1] = 0

    return (det_hessian * 255).astype(np.uint8)


# ransac line fit
def random_n_points_pair(points: np.ndarray, n: int) -> np.ndarray:
    idx = np.random.choice(points.shape[0], n, replace=False)
    return np.array(list(itertools.combinations(points[idx], 2)))


def fit_line(points: List[np.ndarray]) -> Tuple[float, float, float]:
    m = np.vstack(points)
    m = np.hstack([m, np.ones((m.shape[0], 1))])
    eigen_values, eigen_vectors = np.linalg.eig(m.T @ m)
    index = np.argmin(eigen_values)
    a, b, c = eigen_vectors[:, index]
    norm = (a**2 + b**2) ** 0.5
    a, b, c = a / norm, b / norm, c / norm
    return a, b, c


def distance_to_line(p: np.ndarray, a: float, b: float, c: float):
    x, y = p
    return abs(a * x + b * y + c)


def ransac_one_iter(
    all_points: np.ndarray, num_point_pairs: int = 1000, threshold: float = 2
):
    point_pairs = random_n_points_pair(all_points, num_point_pairs)

    # find the line with most inliers
    best_inlier_count = 0
    best_line_inliers = None
    for cur_pair in tqdm(point_pairs):
        a, b, c = fit_line([cur_pair[0], cur_pair[1]])
        cur_inliers = [
            p for p in all_points if distance_to_line(p, a, b, c) < threshold
        ]
        inlier_count = len(cur_inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line_inliers = cur_inliers

    # refit the line with all inliers
    a, b, c = fit_line(best_line_inliers)  # type: ignore
    all_points = np.array(
        [p for p in all_points if distance_to_line(p, a, b, c) >= threshold]
    )
    return a, b, c, all_points, len(best_line_inliers), best_line_inliers  # type: ignore


def ransac(
    all_points: np.ndarray,
    num_lines: int,
    num_point_pairs: int = 300,
    threshold: float = 2,
):
    all_lines = []
    for _ in range(num_lines):
        a, b, c, all_points, num_inliers, inliers = ransac_one_iter(
            all_points, num_point_pairs, threshold
        )
        all_lines.append([a, b, c, num_inliers, inliers])
    return all_lines


def get_y_from_x(
    x: Union[float, np.ndarray], a: float, b: float, c: float
) -> Union[float, np.ndarray]:
    return -(a * x + c) / b


# hough transform
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
            rho = x[i] * np.cos(thetas[j]) + y[i] * np.sin(thetas[j])
            rho = int(round(((rho + diag_dist) // rho_step)))
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
    theta_step: float = 1.0,
    rho_step: float = 1.0,
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


# functionalities
@app.command("corner-keypoint", help="Detect corner keypoints in an image")
def corner_keypoint(
    input_image_path: str = typer.Option(
        os.path.join("data/road.png"),
        "--input",
        "-i",
        help="Input image path",
    ),
    output_folder_path: str = typer.Option(
        os.path.join("result"), "--output", "-o", help="Output folder path"
    ),
    sigma: float = typer.Option(
        6.0, "--sigma", "-sig", help="Sigma (variance in gaussian filer)"
    ),
    gaussian_filter_size: Union[int, None] = typer.Option(
        None,
        "--gaussian-size",
        "-gs",
        help="Size of the filter, should be at least less than image size"
        ", if it is not specified, it will be set to floor(6 * sigma + 1))",
    ),
    threshold: Union[float, None] = typer.Option(
        16.0,
        "--threshold",
        "-t",
        help="Threshold for corner keypoints as float, either input a value or use quantile",
    ),
    threshold_quantile: Union[float, None] = typer.Option(
        None, help="Threshold quantile"
    ),
):
    # param check
    if threshold is None:
        if threshold_quantile is None:
            raise ValueError("Either threshold or threshold quantile must be specified")
    elif threshold_quantile is not None:
        raise ValueError("Only one of threshold or threshold quantile can be specified")
    # load image
    image = load_image(input_image_path)
    # detect corner keypoints
    result = corner_keypoint_detector(
        image=image,
        sigma=sigma,
        gaussian_filter_size=gaussian_filter_size,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
    )
    # save result
    cv2.imwrite(os.path.join(output_folder_path, "keypoint.png"), result)


@app.command("ransac-linefit", help="Detect lines in an image using RANSAC")
def ransac_fit_func(
    input_image_path: str = typer.Option(
        os.path.join("result/keypoint.png"),
        "--input",
        "-i",
        help="Input keypoint image path",
    ),
    output_folder_path: str = typer.Option(
        os.path.join("result"), "--output", "-o", help="Output folder path"
    ),
    num_lines: int = typer.Option(
        4, "--num-lines", "-n", help="Number of lines to fit"
    ),
    num_point_samples: int = typer.Option(
        100,
        "--num-point-samples",
        "-ns",
        help="Number of point samples. Their combination will be point pairs space",
    ),
    threshold: float = typer.Option(
        2.0, "--threshold", "-t", help="Threshold for inliers"
    ),
) -> None:
    # load image and keypoints
    image: np.ndarray = cv2.imread(input_image_path, 0)
    y, x = np.where(image > 0)
    x, y = x.astype(np.float32), y.astype(
        np.float32
    )  # make sure we are in standard Cartesian coordinates

    # ransac
    all_points = np.stack([x, y], axis=1)
    result = ransac(
        all_points,
        num_lines=num_lines,
        num_point_pairs=num_point_samples,
        threshold=threshold,
    )

    # plot
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for cur_r in result:
        a, b, c, num_support, inliers = cur_r
        x, y = np.stack(inliers, axis=1)
        for point in inliers:
            pt1 = (int(point[0]) - 1, int(point[1]) - 1)
            pt2 = (int(point[0]) + 1, int(point[1]) + 1)
            cv2.rectangle(image, pt1, pt2, (0, 255, 255), -1)
        x0, y0 = int(x.min()), int((-a * x.min() - c) / b)
        x1, y1 = int(x.max()), int((-a * x.max() - c) / b)
        cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), thickness=2)
        text_pos = (x0 + 20, y0 + 10)
        cv2.putText(
            image,
            f"{num_support}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )
    cv2.imwrite(os.path.join(output_folder_path, "ransac.png"), image)


@app.command("hough-transform", help="Detect lines in an image using Hough Transform")
def hough_line_fit(
    input_image_path: str = typer.Option(
        os.path.join("result/keypoint.png"),
        "--input",
        "-i",
        help="Input keypoint image path",
    ),
    output_folder_path: str = typer.Option(
        os.path.join("result"), "--output", "-o", help="Output folder path"
    ),
    num_lines: int = typer.Option(
        4, "--num-lines", "-n", help="Number of lines to fit"
    ),
    theta_step: float = typer.Option(
        2.0, "--theta-step", "-ts", help="Theta step size"
    ),
    rho_step: float = typer.Option(2.0, "--rho-step", "-rs", help="Rho step size"),
    min_dist: float = typer.Option(
        10.0,
        "--min-dist",
        "-md",
        help="Distance to determine which points are inliers after line fitting",
    ),
) -> None:
    # load keypoint image
    image = cv2.imread(input_image_path, 0)

    # hough transform
    detected_lines, accumulators, inliers = hough_transform_line_detection(
        keypoint_image=image,
        theta_step=theta_step,
        rho_step=rho_step,
        min_dist=min_dist,
        num_lines=num_lines,
    )

    # plot results
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
            os.path.join(
                output_folder_path,
                f"hough_accumulator_theta_step_{theta_step}_rho_step_{rho_step}_line_{cur_index}.png",
            ),
            scaled_accumulator,  # type: ignore
        )  # type: ignore
    cv2.imwrite(
        os.path.join(
            output_folder_path, f"hough_theta_step_{theta_step}_rho_step_{rho_step}.png"
        ),
        image,
    )


if __name__ == "__main__":
    app()

import os
import cv2
import typer
import warnings
import itertools
import numpy as np
from rich import print
from typing import Union, Tuple, List
from PIL import Image
from numba import jit
from tqdm.auto import tqdm


app = typer.Typer()
np.random.seed(42)
warnings.filterwarnings("ignore")


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
def generate_sample_space(
    num_samples: int, num_points_per_sample: int, keypoints: List[Tuple[int, int]]
) -> List[Tuple[Tuple[int, int]]]:
    # sample index
    sampled_groups_index = [
        tuple(
            np.random.choice(len(keypoints), size=num_points_per_sample, replace=False)
        )
        for _ in range(num_samples)
    ]
    while len(set(sampled_groups_index)) != len(sampled_groups_index):
        sampled_groups_index = [
            tuple(
                np.random.choice(
                    len(keypoints), size=num_points_per_sample, replace=False
                )
            )
            for _ in range(num_samples)
        ]
    # convert index to points
    sampled_groups = []
    for one_group in sampled_groups_index:
        sampled_groups.append(tuple(keypoints[i] for i in one_group))
    return sampled_groups


@jit(nopython=True)
def fit_line(points: List[Tuple[int, int]]) -> np.ndarray:
    # assume the line is ax + by + c = 0
    points = np.array(points, dtype=np.float64)  # type: ignore
    points = np.hstack((points, np.ones((points.shape[0], 1))))  # type: ignore
    eigen_values, eigen_vectors = np.linalg.eig(points.T @ points)  # type: ignore
    index = np.argmin(eigen_values)
    return eigen_vectors[:, index]


@jit(nopython=True)
def distance_to_a_line(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.hstack((points, np.ones((points.shape[0], 1))))  # type: ignore
    return np.abs(np.sum(points * coefficients, axis=1)) / np.sqrt(
        coefficients[0] ** 2 + coefficients[1] ** 2
    )


def compute_y(x: int, a: float, b: float, c: float) -> int:
    return int((-a * x - c) / b)


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
        4.0, "--sigma", "-sig", help="Sigma (variance in gaussian filer)"
    ),
    gaussian_filter_size: Union[int, None] = typer.Option(
        None,
        "--gaussian-size",
        "-gs",
        help="Size of the filter, should be at least less than image size"
        ", if it is not specified, it will be set to floor(6 * sigma + 1))",
    ),
    threshold: Union[float, None] = typer.Option(
        30.0,
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
        10, "--num-lines", "-n", help="Number of lines to fit"
    ),
    num_iterations: int = typer.Option(
        1000, "--num-iterations", "-it", help="Number of iterations per line fit"
    ),
    num_sample: int = typer.Option(
        2, "--num-sample", "-s", help="Number of point samples per line fit"
    ),
    confidence_val: int = typer.Option(
        100,
        "--confidence",
        "-c",
        help="Confidence the number of inliers to support the line",
    ),
    threshold: int = typer.Option(
        5, "--threshold", "-t", help="Threshold for inliers"
    ),
    max_couner: int = typer.Option(
        200,
        "--max-counter",
        "-mc",
        help="Max number of time to sample from all points",
    ),
) -> None:
    # load image
    image = load_image(input_image_path)
    # get all keypoints coordinates
    keypoints = np.argwhere(image == 255.0).astype(np.float64)
    keypoints = [tuple(keypoint) for keypoint in keypoints]
    c_limit = image.shape[1] / 3
    keypoints = [keypoint for keypoint in keypoints if keypoint[1] < (1.5 * c_limit)]
    # run ransac
    line_counter = 0
    results = []
    counter = 0
    while (line_counter < num_lines) and (counter < max_couner):
        # generate sample space
        sampled_groups = generate_sample_space(
            num_samples=num_iterations, num_points_per_sample=num_sample, keypoints=keypoints  # type: ignore
        )
        for cur_point_set in tqdm(sampled_groups):
            # get remaining points
            remaining_points = np.array(list(set(keypoints) - set(cur_point_set)))
            # fit line
            coefs = fit_line(points=cur_point_set)  # type: ignore
            # get distance to the line
            distances = distance_to_a_line(coefficients=coefs, points=remaining_points)
            # check number of inliners
            if np.sum(distances < threshold) >= confidence_val:
                # if enough inliners, fit the line again with all inliners and record result
                inliners = np.vstack(
                    (remaining_points[distances < threshold], np.array(cur_point_set))
                )
                inliners = [tuple(i) for i in inliners]
                print(len(inliners))
                coefs = fit_line(points=inliners)  # type: ignore
                results.append((coefs, inliners))
                keypoints = list(set(keypoints) - set(inliners))
                line_counter += 1
                counter = 0
                break
        counter += 1
    print(f"Found {len(results)} lines")
    # plot results
    for r in results:
        coefs, points = r
        a, b, c = coefs[0], coefs[1], coefs[2]
        line_start = (0, compute_y(0, a, b, c))
        line_end = (image.shape[1] - 1, compute_y(image.shape[1] - 1, a, b, c))
        cv2.line(image, line_start, line_end, 255, 1)  # type: ignore

    cv2.imwrite(os.path.join(output_folder_path, "line.png"), image)


if __name__ == "__main__":
    app()

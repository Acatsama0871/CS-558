import os
import cv2
import typer
import itertools
import numpy as np
from rich import print
from typing import Union, Tuple
from PIL import Image


app = typer.Typer()


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
    else:
        if threshold_quantile is not None:
            raise ValueError(
                "Only one of threshold or threshold quantile can be specified"
            )
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
            else:
                if det_hessian[x, y] != 0:  # if all zeros, keep it
                    det_hessian[x - 1 : x + 2, y - 1 : y + 2] = 0  # type: ignore
                    det_hessian[x, y] = 1
    # ignore border
    det_hessian[0, :] = 0
    det_hessian[-1, :] = 0
    det_hessian[:, 0] = 0
    det_hessian[:, -1] = 0

    return (det_hessian * 255).astype(np.uint8)


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
        2.0, "--sigma", "-sig", help="Sigma (variance in gaussian filer)"
    ),
    gaussian_filter_size: Union[int, None] = typer.Option(
        None,
        "--gaussian-size",
        "-gs",
        help="Size of the filter, should be at least less than image size"
        ", if it is not specified, it will be set to floor(6 * sigma + 1))",
    ),
    threshold: Union[float, None] = typer.Option(
        25.0,
        "--threshold",
        "-t",
        help="Threshold for corner keypoints as float, either input a value or use quantile",
    ),
    threshold_quantile: Union[float, None] = typer.Option(
        None, help="Threshold quantile"
    ),
    cross_size: int = typer.Option(2, help="Size of the cross"),
):
    # param check
    if threshold is None:
        if threshold_quantile is None:
            raise ValueError("Either threshold or threshold quantile must be specified")
    else:
        if threshold_quantile is not None:
            raise ValueError(
                "Only one of threshold or threshold quantile can be specified"
            )
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
    # plot result
    y_coords, x_coords = np.where(result > 0)
    # save result
    cv2.imwrite(os.path.join(output_folder_path, "keypoint.png"), result)


if __name__ == "__main__":
    app()

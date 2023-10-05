import os
import typer
import itertools
import numpy as np
from typing import Union, Tuple
from PIL import Image
from rich import print


app = typer.Typer()


# helper function
def load_image(path: str) -> np.array:
    img = Image.open(path)
    img_gray = img.convert("L")
    return np.array(img_gray).astype(np.float64)


def save_image(path: str, image: np.array):
    img = Image.fromarray(image)
    img.convert("L").save(path)


def convolve2d(image: np.array, kernel: np.array) -> np.array:
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


# question 1
def gaussian_func(x: int, y: int, sigma: float, size: int) -> float:
    return (
        1
        / (2 * np.pi * sigma**2)
        * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2))
    )


def gaussian_filter(
    image: np.array, sigma: float, size: Union[int, None] = None
) -> Tuple[np.array, int]:
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


def sobel_operator(image: np.array) -> Tuple[np.array, np.array]:
    # sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # convolve
    return convolve2d(image, sobel_x), convolve2d(image, sobel_y)


@app.command(
    name="gaussian-filter",
    rich_help_panel="Filter",
    help="Gaussian filter with stride 1",
)
def gaussian_filter_func(
    input_image_path: str = typer.Option(
        os.path.join("data/kangaroo.pgm"), "--input", "-i", help="Input image path"
    ),
    output_folder_path: str = typer.Option(
        os.path.join("result"), "--output", "-o", help="Output folder path"
    ),
    sigma: float = typer.Option(
        1.0, "--sigma", "-s", help="Sigma (variance in gaussian filer)"
    ),
    size: Union[int, None] = typer.Option(
        None,
        "--size",
        "-s",
        help="Size of the filter, should be at least less than image size"
        ", if it is not specified, it will be set to floor(6 * sigma + 1))",
    ),
):
    # load image
    image = load_image(input_image_path)
    # convolve
    result, size = gaussian_filter(image=image, sigma=sigma, size=size)
    # save image
    save_image(
        os.path.join(
            output_folder_path,
            f"{input_image_path.split('/')[-1].split('.')[0]}_gaussian_{sigma}_{size}.pgm",
        ),
        result,
    )


@app.command(
    "sobel-operator-gradient-magnitude", rich_help_panel="Filter", help="Sobel operator"
)
def sobel_operator_gradient_magnitude(
    input_image_path: str = typer.Option(
        os.path.join("result/kangaroo_gaussian_1.0_7.pgm"),
        "--input",
        "-i",
        help="Input image path",
    ),
    output_folder_path: str = typer.Option(
        os.path.join("result"), "--output", "-o", help="Output folder path"
    ),
    threshold: float = typer.Option(75.0, "--threshold", "-t", help="Threshold"),
):
    # load image
    image = load_image(input_image_path)
    # convolve
    result_x, result_y = sobel_operator(image=image)
    # calculate magnitude
    magnitude = np.sqrt(result_x**2 + result_y**2)
    magnitude_filter = magnitude > threshold
    filtered_magnitude = magnitude * magnitude_filter
    # save image
    save_image(
        os.path.join(
            output_folder_path,
            f"sobel_magnitude_x_{threshold}_{input_image_path.split('/')[-1].replace('.pgm', '')}.pgm",
        ),
        filtered_magnitude,
    )


@app.command(name="test-load", rich_help_panel="Test", help="Test load image")
def test_load(
    input_image_path: str = typer.Option(
        os.path.join("data/kangaroo.pgm"), "--input", "-i", help="Input image path"
    ),
):
    image = load_image(input_image_path)
    print(image)
    print(image.shape)


@app.command(name="test-save", rich_help_panel="Test", help="Test save image")
def test_save(
    input_image_path: str = typer.Option(
        os.path.join("data/kangaroo.pgm"), "--input", "-i", help="Input image path"
    )
):
    image = load_image(input_image_path)
    save_image("test.pgm", image)


@app.command(name="test-convolve", rich_help_panel="Test", help="Test convolve")
def test_convolve():
    test_input = np.array(
        [[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    test_kernel = (1 / 9) * np.ones((3, 3))
    result = convolve2d(test_input, test_kernel)
    print()
    print(test_input)
    print()
    print(result)


if __name__ == "__main__":
    app()

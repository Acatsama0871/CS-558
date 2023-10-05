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


# gaussian filter
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


# sobel operator
def sobel_operator(image: np.array) -> Tuple[np.array, np.array]:
    # sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # convolve
    return convolve2d(image, sobel_x), convolve2d(image, sobel_y)


# non maximum suppression
def map_to_closest_val(x: np.array, candidates: np.array) -> np.array:
    dists = np.abs(x[:, np.newaxis] - candidates[:, np.newaxis])
    return candidates[np.argmin(dists, axis=1)]


def _direction_zero(
    filtered_intensity: np.array, edge_map: np.array, i: int, j: int
) -> np.array:
    # at the left edge
    if j < 1:
        if filtered_intensity[i, j] >= filtered_intensity[i, j + 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i, j + 1] = 0
    # at the right edge
    elif j >= (edge_map.shape[1] - 1):
        if filtered_intensity[i, j] >= filtered_intensity[i, j - 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i, j] = 1
            edge_map[i, j - 1] = 0
    # in the middle
    elif filtered_intensity[i, j] >= max(
        filtered_intensity[i, j - 1], filtered_intensity[i, j + 1]
    ):
        edge_map[i, j] = filtered_intensity[i, j]
        edge_map[i, j - 1] = 0
        edge_map[i, j + 1] = 0

    return edge_map


def _direction_45(
    filtered_intensity: np.array, edge_map: np.array, i: int, j: int
) -> np.array:
    # 1. at the left edge
    if j < 1:
        # 1.1 at the bottom left corner
        if i == (filtered_intensity.shape[0] - 1):
            edge_map[i, j] = 0
        elif filtered_intensity[i, j] >= filtered_intensity[i + 1, j + 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i + 1, j + 1] = 0
    # 2. at the right edge
    elif j >= (edge_map.shape[1] - 1):
        # 2.1 at the top right corner
        if i == 0:
            edge_map[i, j] = 0
        elif filtered_intensity[i, j] >= filtered_intensity[i - 1, j - 1]:
            #edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i, j] = 1
            edge_map[i - 1, j - 1] = 0
    elif i < 1:
        # 3.1 at the top left corner
        if j == 0:
            edge_map[i, j] = 0
        # 3.2 at the top edge middle
        elif filtered_intensity[i, j] >= filtered_intensity[i + 1, j + 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i + 1, j + 1] = 0
    # 4. at the bottom edge
    elif i >= (edge_map.shape[0] - 1):
        # 4.1 at the bottom right corner
        if j == (filtered_intensity.shape[1] - 1):
            edge_map[i, j] = 0
        # 4.2 at the bottom edge middle
        elif filtered_intensity[i, j] >= filtered_intensity[i - 1, j - 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i - 1, j - 1] = 0
    # 5. middle
    elif filtered_intensity[i, j] >= max(
        filtered_intensity[i - 1, j - 1], filtered_intensity[i + 1, j + 1]
    ):
        edge_map[i, j] = filtered_intensity[i, j]
        edge_map[i - 1, j - 1] = 0
        edge_map[i + 1, j + 1] = 0
    return edge_map


def _direction_90(
    filtered_intensity: np.array, edge_map: np.array, i: int, j: int
) -> np.array:
    # 1. at the top edge
    if i < 1:
        if filtered_intensity[i, j] >= filtered_intensity[i + 1, j]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i + 1, j] = 0
    elif i >= (edge_map.shape[0] - 1):
        if filtered_intensity[i, j] >= filtered_intensity[i - 1, j]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i - 1, j] = 0
    elif filtered_intensity[i, j] >= max(
        filtered_intensity[i - 1, j], filtered_intensity[i + 1, j]
    ):
        edge_map[i, j] = filtered_intensity[i, j]
        edge_map[i - 1, j] = 0
        edge_map[i + 1, j] = 0
    return edge_map


def _direction_135(
    filtered_intensity: np.array, edge_map: np.array, i: int, j: int
) -> np.array:
    # 1. at the left edge
    if j < 1:
        # 1.1 at the top left corner
        if i == 0:
            edge_map[i, j] = 0
        # 1.2 at the left edge middle
        elif filtered_intensity[i, j] >= filtered_intensity[i - 1, j + 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i - 1, j + 1] = 0
    # 2. at the right edge
    elif j >= (edge_map.shape[1] - 1):
        # 1.1 at the bottom right corner
        if i == (filtered_intensity.shape[0] - 1):
            edge_map[i, j] = 0
        # 1.2 at the right edge middle
        elif filtered_intensity[i, j] >= filtered_intensity[i + 1, j - 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i + 1, j - 1] = 0
    # 3. at the top edge
    elif i < 1:
        # 3.1 at the top left corner
        if j == 0:
            edge_map[i, j] = 0
        # 3.2 at the top edge middle
        elif filtered_intensity[i, j] >= filtered_intensity[i + 1, j - 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i + 1, j - 1] = 0
    # 4. at the bottom edge
    elif i >= (edge_map.shape[0] - 1):
        # 4.1 at the bottom right corner
        if j == (filtered_intensity.shape[1] - 1):
            edge_map[i, j] = 0
        # 4.2 at the bottom edge middle
        elif filtered_intensity[i, j] >= filtered_intensity[i - 1, j + 1]:
            edge_map[i, j] = filtered_intensity[i, j]
            edge_map[i - 1, j + 1] = 0
    # 5. middle
    elif filtered_intensity[i, j] >= max(
        filtered_intensity[i - 1, j + 1], filtered_intensity[i + 1, j - 1]
    ):
        edge_map[i, j] = filtered_intensity[i, j]
        edge_map[i - 1, j + 1] = 0
        edge_map[i + 1, j - 1] = 0
    return edge_map


def non_maximum_suppression_edge_detection(
    sobel_x: np.array, sobel_y: np.array, threshold: float
) -> np.array:
    # direction map
    direction_map = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    direction_map = np.where(direction_map < 0, direction_map + 180, direction_map)
    direction_map = map_to_closest_val(direction_map, np.array([0, 45, 90, 135]))
    # calculate the intensity map
    intensity_map = np.sqrt(sobel_x**2 + sobel_y**2)
    intensity_filter = intensity_map > threshold
    filtered_intensity = intensity_map * intensity_filter
    # edge map
    edge_map = np.zeros_like(filtered_intensity)
    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            if direction_map[i, j] == 0:
                edge_map = _direction_zero(filtered_intensity, edge_map, i, j)
            elif direction_map[i, j] == 45:
                edge_map = _direction_45(filtered_intensity, edge_map, i, j)
            elif direction_map[i, j] == 90:
                edge_map = _direction_90(filtered_intensity, edge_map, i, j)
            elif direction_map[i, j] == 135:
                edge_map = _direction_135(filtered_intensity, edge_map, i, j)

    edge_map = edge_map.astype(np.uint8)
    return edge_map


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
            f"gaussian_{input_image_path.split('/')[-1].split('.')[0]}_{sigma}_{size}.pgm",
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


@app.command(
    "non-maximum-suppression-edge-detection",
    rich_help_panel="Edge",
    help="Non maximum suppression edge detection",
)
def non_maximum_edge_detection_func(
    input_image_path: str = typer.Option(
        os.path.join("data/plane.pgm"), "--input", "-i", help="Input image path"
    ),
    output_folder_path: str = typer.Option(
        os.path.join("result"), "--output", "-o", help="Output folder path"
    ),
    sigma: float = typer.Option(
        6., "--sigma", "-s", help="Sigma (variance in gaussian filer)"
    ),
    gaussian_filter_size: Union[int, None] = typer.Option(
        None,
        "--gaussian-size",
        "-gs",
        help="Size of the filter, should be at least less than image size"
        ", if it is not specified, it will be set to floor(6 * sigma + 1))",
    ),
    threshold: float = typer.Option(22.0, "--threshold", "-t", help="Threshold"),
):
    # load image
    image = load_image(input_image_path)
    # smooth image
    result, size = gaussian_filter(image=image, sigma=sigma, size=gaussian_filter_size)
    # sobel operator
    result_x, result_y = sobel_operator(image=result)
    # non maximum suppression
    edge_map = non_maximum_suppression_edge_detection(
        sobel_x=result_x, sobel_y=result_y, threshold=threshold
    )
    edge_map = edge_map * 225
    img = Image.fromarray(edge_map)
    img.save(os.path.join("result", f"edge_{input_image_path.split('/')[-1].split('.')[0]}_{sigma}_{size}_{threshold}.png"))


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

import itertools
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Union
from tqdm import tqdm


def convolve2d(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    image = np.array(image)  # type: ignore
    kernel = np.array(kernel)  # type: ignore

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
    image = np.pad(image, (half_k_h, half_k_w), mode="edge")  # type: ignore

    # convolution operation
    output = image.copy()
    for x, y in tqdm(
        itertools.product(
            range(half_k_h, image.shape[0] - half_k_h),
            range(half_k_w, image.shape[1] - half_k_w),
        ),
        total=(image.shape[0] - 2 * half_k_h) * (image.shape[1] - 2 * half_k_w),
        desc="Convolution...",
    ):
        if ((x + half_k_h) > image.shape[0]) or ((y + half_k_w) > image.shape[1]):
            continue
        output[x, y] = (
            kernel
            * image[
                (x - half_k_h) : (x + half_k_h + 1), (y - half_k_w) : (y + half_k_w + 1)
            ]
        ).sum()

    return jnp.array(output[half_k_h:-half_k_h, half_k_w:-half_k_w])


def gaussian_func(x: int, y: int, sigma: float, size: int) -> jnp.ndarray:
    return (
        1
        / (2 * jnp.pi * sigma**2)
        * jnp.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2))
    )


def gaussian_kernel(sigma: float = 1) -> jnp.ndarray:
    size = int(jnp.floor(6 * sigma + 1).item())
    # get gaussian kernel
    kernel = jnp.zeros((size, size))
    for i, j in itertools.product(range(size), range(size)):
        kernel = kernel.at[i, j].set(
            gaussian_func(x=i, y=j, sigma=sigma, size=size).item()
        )
    return kernel / kernel.sum()


_sobel_x_filter = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
_sobel_y_filter = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def sobel_filter(image: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return (
        convolve2d(image, _sobel_x_filter),
        convolve2d(image, _sobel_y_filter),
    )


def non_maximum_suppression(
    response_image: jnp.ndarray, nms_window_size: int = 10
) -> jnp.ndarray:
    result_image = np.zeros_like(response_image)
    for x, y in tqdm(
        itertools.product(
            range(0, response_image.shape[0], nms_window_size),
            range(0, response_image.shape[1], nms_window_size),
        ),
        total=(response_image.shape[0] // nms_window_size)
        * (response_image.shape[1] // nms_window_size),
        desc="Non-Maximum Suppression...",
    ):
        if not jnp.all(
            response_image[x : x + nms_window_size, y : y + nms_window_size] == 0
        ).item():
            cur_window = response_image[
                x : x + nms_window_size, y : y + nms_window_size
            ]
            max_x, may_y = np.unravel_index(np.argmax(cur_window), cur_window.shape)
            result_image[x + max_x, y + may_y] = response_image[x + max_x, y + may_y]

    return jnp.array(result_image)


def harris_corner_detector(
    input_img: jnp.ndarray,
    num_keypoints: int = 1000,
    gaussian_sigma: float = 1,
    harris_response_alpha: float = 0.05,
    nms_window_size: Union[int, None] = None,
    return_raw_feature_map: bool = False,
) -> jnp.ndarray:
    # get gradients
    i_x, i_y = sobel_filter(input_img)
    i_xx = convolve2d(i_x * i_x, gaussian_kernel(gaussian_sigma))
    i_yy = convolve2d(i_y * i_y, gaussian_kernel(gaussian_sigma))
    i_xy = convolve2d(i_x * i_y, gaussian_kernel(gaussian_sigma))

    # harris corner response
    det = i_xx * i_yy - i_xy * i_xy
    trace = i_xx + i_yy
    harris_response = det - harris_response_alpha * trace * trace
    harris_response = jnp.where(harris_response > 0, harris_response, 0)
    if nms_window_size is not None:
        harris_response = non_maximum_suppression(
            response_image=harris_response, nms_window_size=nms_window_size
        )

    # get keypoints
    num_activate_point = jnp.sum(harris_response > 0).item()
    harris_response_flatten = harris_response.flatten()
    if num_activate_point < num_keypoints:
        max_response_idx = jnp.argsort(harris_response_flatten)[-num_activate_point:]
    else:
        max_response_idx = jnp.argsort(harris_response_flatten)[-num_keypoints:]
    maximum_response = jnp.max(harris_response_flatten[max_response_idx]).item()
    return_harris_response_plot = jnp.zeros(harris_response_flatten.shape)
    return_harris_response_plot = return_harris_response_plot.at[max_response_idx].set(
        225
    )
    return_harris_response_plot = return_harris_response_plot.reshape(
        harris_response.shape
    )
    if return_raw_feature_map:
        return_harris_response_feature = jnp.zeros(harris_response_flatten.shape)
        return_harris_response_feature = return_harris_response_feature.at[
            max_response_idx
        ].set(harris_response_flatten[max_response_idx])
        # return_harris_response_feature = harris_response.at[max_response_idx].divide(maximum_response)
        # return_harris_response_feature = return_harris_response_feature.at[max_response_idx].multiply(255)
        # return_harris_response_feature = return_harris_response_feature.reshape(harris_response.shape).astype(np.uint8)
        return_harris_response_feature = harris_response.reshape(harris_response.shape)

        return return_harris_response_feature, return_harris_response_plot
    else:
        return return_harris_response_plot.astype(np.uint8)

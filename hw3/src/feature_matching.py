import itertools
import numpy as np
from typing import Tuple, Union
from tqdm.auto import tqdm


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

    return output[half_k_h:-half_k_h, half_k_w:-half_k_w]


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


def sobel_operator(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # convolve
    return convolve2d(image, sobel_x), convolve2d(image, sobel_y)


def non_maximum_suppression(
    response_image: np.ndarray, nms_window_size: int = 10
) -> np.ndarray:
    # Copy the original image to the result image
    result_image = response_image.copy()

    # Get the dimensions of the image
    height, width = response_image.shape

    # Define the half size of the NMS window
    half_window = nms_window_size // 2

    # Iterate over the image (excluding the border pixels)
    for y, x in tqdm(
        itertools.product(
            range(half_window, height - half_window),
            range(half_window, width - half_window),
        ),
        total=(height - 2 * half_window) * (width - 2 * half_window),
        desc="Non-maximum suppression...",
    ):
        # Get the local window around the current pixel
        local_window = response_image[
            y - half_window : y + half_window + 1,
            x - half_window : x + half_window + 1,
        ]

        # Get the maximum value in the local window
        local_max = np.max(local_window)

        # Suppress the pixel if it is not the local maximum
        if response_image[y, x] < local_max:
            result_image[y, x] = 0

    return result_image


def harris_detector(
    image: np.ndarray,
    alpha: float = 0.05,
    num_keypoints: int = 1000,
    apply_non_max_suppression=False,
    gaussian_filter_size: int = 5,
    non_max_suppression_size: int = 3,
    sigma: float = 1,
) -> np.ndarray:
    # gaussian smoothing
    image, _ = gaussian_filter(image=image, sigma=sigma, size=gaussian_filter_size)
    # corner response
    Ix, Iy = sobel_operator(image=image)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixx, _ = gaussian_filter(image=Ixx, sigma=1)
    Iyy, _ = gaussian_filter(image=Iyy, sigma=1)
    Ixy, _ = gaussian_filter(Ixy, sigma=1)
    det_M = (Ixx * Iyy) - (Ixy**2)
    trace_M = Ixx + Iyy
    corner_response = det_M - alpha * (trace_M**2)

    # non-maximum suppression
    if apply_non_max_suppression:
        corner_response = non_maximum_suppression(
            response_image=corner_response,
            nms_window_size=non_max_suppression_size,
        )

    # select top k keypoints
    corners_response_map = np.argpartition(corner_response.flatten(), -num_keypoints)[
        -num_keypoints:
    ]
    corner_response_mask = np.zeros_like(corner_response).reshape(-1)
    corner_response_mask[corners_response_map] = 1
    corner_response_mask = corner_response_mask.reshape(corner_response.shape)
    corner_response = corner_response * corner_response_mask

    # normalize
    corner_response = corner_response / np.max(corner_response)
    corner_response = (corner_response * 255).astype(np.uint8)

    return corner_response


def ssd_similarity(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    return np.sum((descriptor1 - descriptor2) ** 2)  # type: ignore


def ncc_similarity(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    mask1 = descriptor1 > 0
    mask2 = descriptor2 > 0
    descriptor1[mask1] = 255
    descriptor2[mask2] = 255
    descriptor1 = descriptor1 - np.mean(descriptor1)
    descriptor2 = descriptor2 - np.mean(descriptor2)
    return np.sum(descriptor1 * descriptor2) / np.sqrt(
        np.sum(descriptor1**2) * np.sum(descriptor2**2)
    )


def extract_keypoint_descriptor(image: np.ndarray, descriptor_size: int = 3):
    x, y = np.nonzero(image)
    keypoints = []
    descriptors = []
    for i, j in zip(x, y):
        if (
            ((i - descriptor_size) >= 0)
            and ((j - descriptor_size) >= 0)
            and (i + descriptor_size < image.shape[0])
            and (j + descriptor_size < image.shape[1])
        ):
            keypoints.append((i, j))
            descriptors.append(
                image[
                    i - descriptor_size : i + descriptor_size + 1,
                    j - descriptor_size : j + descriptor_size + 1,
                ]
            )
    return keypoints, descriptors


def feature_matching_func(
    response_image_1: np.ndarray,
    response_image_2: np.ndarray,
    top_k: int = 20,
    descriptor_size: int = 3,
    similarity_measure: str = "ssd",
    return_all: bool = False,
):
    # extract keypoints and descriptors
    image1_keypoints, image1_descriptors = extract_keypoint_descriptor(
        response_image_1, descriptor_size
    )
    image2_keypoints, image2_descriptors = extract_keypoint_descriptor(
        response_image_2, descriptor_size
    )

    # matching
    matched_keypoints1 = []
    matched_keypoints2 = []
    matched_scores = []
    used_keypoints2 = []
    p_bar = tqdm(total=len(image1_descriptors) * len(image2_descriptors))
    with p_bar:
        for i in range(len(image1_descriptors)):
            best_match = None
            best_score = (
                float("inf") if (similarity_measure == "ssd") else float("-inf")
            )
            for j in range(len(image2_descriptors)):
                score = (
                    ssd_similarity(image1_descriptors[i], image2_descriptors[j])
                    if (similarity_measure == "ssd")
                    else ncc_similarity(image1_descriptors[i], image2_descriptors[j])
                )
                if (
                    similarity_measure == "nss"
                    and score > best_score
                    or similarity_measure != "nss"
                    and similarity_measure == "ssd"
                    and score < best_score
                ):
                    best_match = j
                    best_score = score
                p_bar.update(1)
            if best_match is not None:
                matched_keypoints1.append(image1_keypoints[i])
                matched_keypoints2.append(image2_keypoints[best_match])  # type: ignore
                matched_scores.append(best_score)
                used_keypoints2.append(best_match)

    # select top k matches
    return_keypoints1 = []
    return_keypoints2 = []
    return_scores = []
    if return_all:
        top_k = len(matched_scores)
    if similarity_measure == "nss":
        index_sort = np.argsort(matched_scores)[::-1][:top_k]
        for i in index_sort:
            return_keypoints1.append(matched_keypoints1[i])
            return_keypoints2.append(matched_keypoints2[i])
            return_scores.append(matched_scores[i])

    elif similarity_measure == "ssd":
        index_sort = np.argsort(matched_scores)[:top_k]
        for i in index_sort:
            return_keypoints1.append(matched_keypoints1[i])
            return_keypoints2.append(matched_keypoints2[i])
            return_scores.append(matched_scores[i])

    return return_keypoints1, return_keypoints2, return_scores

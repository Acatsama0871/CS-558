#!/usr/bin/env python
# coding: utf-8

# In[101]:


import numpy as np
from PIL import Image
import itertools
from typing import Union, Tuple, List
import cv2
import matplotlib.pyplot as plt


# In[102]:


# helper function
def load_image(path: str) -> np.ndarray:
    img = Image.open(path)
    img_gray = img.convert("L")
    return np.array(img_gray).astype(np.float64)


def save_image(path: str, image: np.ndarray):
    img = Image.fromarray(image)
    img.convert("L").save(path)


# In[103]:


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


# In[104]:


# gaussian filter
def gaussian_func(x: int, y: int, sigma: float, size: int) -> float:
    return (1 / (2 * np.pi * sigma**2) * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2)))

def gaussian_filter(image: np.ndarray, sigma: float, size: Union[int, None] = None) -> Tuple[np.ndarray, int]:
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


# In[105]:


# sobel operator
def sobel_operator(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # sobel operator
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # convolve
    return convolve2d(image, sobel_x), convolve2d(image, sobel_y)


# In[106]:


# harris corner detector
def harris_corners(path: str, alpha: float = 0.06, top_points: int = 1000, apply_non_max_suppression=False) -> np.ndarray:
    img = load_image(path)
    # sobel operation
    Ix , Iy = sobel_operator(img)
    # calculations for second momentum matrix (M)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    # gaussian filtering
    Ixx , _ = gaussian_filter(Ixx, sigma = 1)
    Iyy , _ = gaussian_filter(Iyy, sigma = 1)
    Ixy , _ = gaussian_filter(Ixy, sigma = 1)
    # calculating cornerness (R)
    det_M = (Ixx * Iyy) - (Ixy**2)
    trace_M = Ixx + Iyy
    R = det_M - alpha * (trace_M**2)
    
    corners = np.argpartition(R.flatten(), -top_points)[-top_points:]
    corners = np.column_stack(np.unravel_index(corners, R.shape))
    
    if apply_non_max_suppression:
        # applying non-maximum suppression
        corners = non_max_suppression(corners, R)


    return corners


# In[107]:


def non_max_suppression(corners, R):
    sorted_corners = corners[np.argsort(R[corners[:, 0], corners[:, 1]])[::-1]]

    result = []
    while len(sorted_corners) > 0:
        current_corner = sorted_corners[0]
        result.append(current_corner)

        distances = np.sqrt(np.sum((sorted_corners - current_corner)**2, axis=1))
        mask = distances > 5 
        sorted_corners = sorted_corners[mask]

    return np.array(result)


# In[108]:


def compute_ssd(p1, p2):
    return np.sum((p1 - p2)**2)


# In[117]:


def find_best_matches(left_features, right_features, left_image, right_image, num_matches: int = 20):
    best_matches = []

    for left_feature in left_features:
        left_patch = left_image[
            left_feature[0] - 25 : left_feature[0] + 26,
            left_feature[1] - 25 : left_feature[1] + 26
        ]

        ssd_values = []

        for right_feature in right_features:
            right_patch = right_image[
                right_feature[0] - 25 : right_feature[0] + 26,
                right_feature[1] - 25 : right_feature[1] + 26
            ]
            if (left_patch.shape != (51, 51)) or (right_patch.shape != (51, 51)):
                continue
            else:
                print(left_patch.shape, right_patch.shape)
                ssd = compute_ssd(left_patch, right_patch)
                ssd_values.append(ssd)
                print(ssd_values)

        best_match_index = np.argmin(ssd_values)
        best_matches.append((left_feature, right_features[best_match_index]))

    return best_matches[:num_matches]


# In[110]:


img_left = load_image("uttower_left.jpg")
img_right = load_image("uttower_right.jpg")

harris1 = harris_corners("uttower_left.jpg")
harris2 = harris_corners("uttower_right.jpg")


# In[111]:


harris_left = np.zeros_like(img_left, dtype=np.uint8)
harris_left[harris1[:, 0], harris1[:, 1]] = 255

harris_right = np.zeros_like(img_right, dtype=np.uint8)
harris_right[harris2[:, 0], harris2[:, 1]] = 255


# In[112]:


save_image("left_harris.jpg",harris_left)
save_image("right_harris.jpg",harris_right)


# In[113]:


harris3 = harris_corners("uttower_left.jpg", apply_non_max_suppression=True)
harris4 = harris_corners("uttower_right.jpg", apply_non_max_suppression=True)


# In[114]:


harris_left_nms = np.zeros_like(img_left, dtype=np.uint8)
harris_left_nms[harris3[:, 0], harris3[:, 1]] = 255

harris_right_nms = np.zeros_like(img_right, dtype=np.uint8)
harris_right_nms[harris4[:, 0], harris4[:, 1]] = 255


# In[115]:


save_image("left_harris_nms.jpg",harris_left_nms)
save_image("right_harris_nms.jpg",harris_right_nms)


# In[118]:


similarity_SSD = find_best_matches(harris3,harris4,img_left,img_right)

result_image = np.concatenate((img_left, img_right), axis=1)

for left_feature, right_feature in similarity_SSD:
    pt1 = (left_feature[1], left_feature[0])
    pt2 = (right_feature[1] + img_left.shape[1], right_feature[0])
    cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)
    
save_image("similarity_SSD.jpg",result_image)


import itertools
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Union, List
from tqdm import tqdm


@jax.jit
def ssd_similarity(descriptor1: jnp.ndarray, descriptor2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((descriptor1 - descriptor2) ** 2)


@jax.jit
def ncc_similarity(descriptor1: jnp.ndarray, descriptor2: jnp.ndarray) -> jnp.ndarray:
    descriptor1 = descriptor1 - jnp.mean(descriptor1)
    descriptor2 = descriptor2 - jnp.mean(descriptor2)
    return jnp.sum(descriptor1 * descriptor2) / jnp.sqrt(
        jnp.sum(descriptor1**2) * jnp.sum(descriptor2**2)
    )


def extract_keypoint_descriptor(
    response_map: jnp.ndarray, descriptor_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x_idx, y_idx = jnp.nonzero(response_map)

    descriptor_list = []
    for x, y in tqdm(
        zip(x_idx, y_idx), total=len(x_idx), desc="Extracting descriptors..."
    ):
        descriptor = response_map[
            x - descriptor_size // 2 : x + descriptor_size // 2 + 1,
            y - descriptor_size // 2 : y + descriptor_size // 2 + 1,
        ]
        if (descriptor.shape[0] == 11) and (  # TODO: dynamic descriptor size
            descriptor.shape[1] == 11
        ):
            descriptor_list.append(descriptor)
    return jnp.stack(descriptor_list), jnp.hstack(
        [x_idx.reshape(-1, 1), y_idx.reshape(-1, 1)]
    )


def match_features(
    response_map1: jnp.ndarray,
    response_map2: jnp.ndarray,
    descriptor_size: int,
    similarity: str,
    return_size: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # similarity measures
    match similarity:
        case "ssd":
            similarity_func = ssd_similarity
        case "ncc":
            similarity_func = ncc_similarity
        case _:
            raise ValueError("Invalid similarity function")

    # extract descriptors
    descriptor1, keypoint1 = extract_keypoint_descriptor(response_map1, descriptor_size)
    descriptor2, keypoint2 = extract_keypoint_descriptor(response_map2, descriptor_size)

    # compute similarity matrix
    matched_keypoints1 = []
    matched_keypoints2 = []
    matched_scores = []
    used_keypoints2 = []
    for i in tqdm(range(descriptor1.shape[0]), desc="Computing similarity..."):
        best_match = None
        best_score = float("inf") if (similarity == "ssd") else float("-inf")
        for j in range(descriptor2.shape[0]):
            score = similarity_func(descriptor1[i], descriptor2[j])
            if (
                (similarity == "ssd")
                and score < best_score
                or similarity != "ssd"
                and score > best_score
            ):
                if j not in used_keypoints2:
                    best_score = score
                    best_match = j
        matched_keypoints1.append(keypoint1.at[i].get())
        matched_keypoints2.append(keypoint2.at[best_match].get())
        matched_scores.append(best_score)
        used_keypoints2.append(best_match)

    # sort the matches
    sorted_matches = np.argsort(matched_scores)[::-1][:return_size]
    matched_keypoints1 = [
        (matched_keypoints1[i][0].item(), matched_keypoints1[i][1].item())
        for i in sorted_matches
    ]
    matched_keypoints2 = [
        (matched_keypoints2[i][0].item(), matched_keypoints2[i][1].item())
        for i in sorted_matches
    ]

    return matched_keypoints1, matched_keypoints2  # type: ignore

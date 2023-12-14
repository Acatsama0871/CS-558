import jax
import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm
from typing import Tuple


@jax.jit
def construct_matrix(point: jnp.ndarray) -> jnp.ndarray:
    return jnp.array(
        [
            [point.at[0].get(), point.at[1].get(), 1, 0, 0, 0],
            [0, 0, 0, point.at[0].get(), point.at[1].get(), 1],
        ]
    )


construct_matrix = jax.vmap(construct_matrix, in_axes=1, out_axes=0)


@jax.jit
def transform_b(point: jnp.ndarray) -> jnp.ndarray:
    return point.T.flatten().reshape(-1, 1)


@jax.jit
def solve_system(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.lstsq(a, b, rcond=None)[0].reshape(2, 3)


@jax.jit
def calculate_dist_with_transformation(
    homography_matrix: jnp.ndarray, point: jnp.ndarray, target_point: jnp.ndarray
) -> jnp.ndarray:
    point = jnp.vstack((point, jnp.ones((1, point.shape[1]))))
    transformed_point = homography_matrix @ point
    return jnp.mean(jnp.linalg.norm(transformed_point - target_point, axis=0))


def ransac_homography(
    img1_keypoints: jnp.ndarray,
    img2_keypoints: jnp.ndarray,
    dist_threshold: float,
    max_iter: int,
    sample_per_iter: int,
    random_seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(random_seed)

    # ransac fit
    for _ in tqdm(range(max_iter)):
        # sample points
        cur_key, key = jax.random.split(key)
        # while True:
        sample_index = jnp.array(
            jnp.unique(
                jax.random.randint(
                    cur_key,
                    (sample_per_iter + 3,),
                    minval=0,
                    maxval=img1_keypoints.shape[1],
                )
            )
        )
        sample_index = sample_index[:sample_per_iter]
        cur_image1_keypoints = img1_keypoints.at[:, sample_index].get()
        cur_image2_keypoints = img2_keypoints.at[:, sample_index].get()
        matrix_a = construct_matrix(cur_image1_keypoints).reshape(-1, 6)
        vector_b = transform_b(cur_image2_keypoints)
        cur_homography_matrix = solve_system(matrix_a, vector_b)
        cur_dist = calculate_dist_with_transformation(
            cur_homography_matrix, img1_keypoints, img2_keypoints
        )
        # stop condition
        if cur_dist < dist_threshold:
            break

    # calculate inliers
    index_set = set(range(img1_keypoints.shape[1]))
    inlier_index = set(np.array(sample_index).tolist())  # type: ignore
    outlier_index = index_set - inlier_index
    outlier_index = np.array(list(outlier_index))
    inlier_keypoints1 = img1_keypoints.at[:, sample_index].get()[:, :, jnp.newaxis]  # type: ignore
    outlier_keypoints1 = img1_keypoints.at[:, outlier_index].get()[:, jnp.newaxis, :]  # type: ignore
    distance_matrix = jnp.linalg.norm(inlier_keypoints1 - outlier_keypoints1, axis=0)
    inlier_mask = distance_matrix <= dist_threshold
    outlier_index = jnp.unique(
        jnp.concatenate([jnp.nonzero(row)[0] for row in inlier_mask])
    )
    inlier_index = jnp.concatenate([sample_index, outlier_index])  # type: ignore
    inlier_image1_keypoints = img1_keypoints.at[:, inlier_index].get()
    inlier_image2_keypoints = img2_keypoints.at[:, inlier_index].get()

    # calculate homography matrix
    matrix_a = construct_matrix(inlier_image1_keypoints).reshape(-1, 6)
    vector_b = transform_b(inlier_image2_keypoints)
    homography_matrix = solve_system(matrix_a, vector_b)

    return homography_matrix, inlier_keypoints1

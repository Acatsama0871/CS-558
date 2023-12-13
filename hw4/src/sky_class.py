import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from typing import Tuple, List
from rich import print
from .k_means import kmeans_fit


def get_pixel_set(
    image: jnp.ndarray, masked_image: jnp.ndarray, mask_color: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    sky_pixel_mask = jnp.all(masked_image == mask_color, axis=-1)
    no_sky_pixel_mask = jnp.logical_not(sky_pixel_mask)
    return image[sky_pixel_mask], image[no_sky_pixel_mask]


def construct_sky_visual_word(
    image: jnp.ndarray,
    masked_image: jnp.ndarray,
    num_cluster: int,
    random_seed: int,
    tol: float,
    max_iter: int,
    mask_color: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # get pixel set
    sky_pixel_set, no_sky_pixel_set = get_pixel_set(image, masked_image, mask_color)
    # k-means
    print("[blue] Finding sky centroids... [/blue]")
    sky_centroids, _ = kmeans_fit(
        x=sky_pixel_set,
        num_clusters=num_cluster,
        random_seed=random_seed,
        max_iter=max_iter,
        tol=tol,
    )
    sky_centroids = jnp.unique(sky_centroids, axis=0)
    print("[blue] Finding no-sky centroids... [/blue]")
    no_sky_centroids, _ = kmeans_fit(
        x=no_sky_pixel_set,
        num_clusters=num_cluster,
        random_seed=random_seed,
        max_iter=max_iter,
        tol=tol,
    )
    no_sky_centroids = jnp.unique(no_sky_centroids, axis=0)

    return sky_centroids, no_sky_centroids  # (num_cluster, 3)


def sky_classification(
    train_image: jnp.ndarray,
    train_masked_image: jnp.ndarray,
    test_images: List[jnp.ndarray],
    num_cluster: int,
    random_seed: int,
    mask_color: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> List[jnp.ndarray]:
    # get visual words
    sky_centroids, no_sky_centroids = construct_sky_visual_word(
        image=train_image,
        masked_image=train_masked_image,
        num_cluster=num_cluster,
        random_seed=random_seed,
        tol=tol,
        max_iter=max_iter,
        mask_color=mask_color,
    )
    no_sky_centroid_mark = sky_centroids.shape[0]
    centroids = jnp.concatenate(
        [sky_centroids, no_sky_centroids], axis=0
    )  # (num_sky_cluster + num_no_sky_cluster, 3)

    # classification
    result_image = []
    for cur_test_image in tqdm(test_images, desc="Classifying test images..."):
        cur_height, cur_width, _ = cur_test_image.shape
        cur_test_image = cur_test_image.reshape(-1, 3)
        # find the shortest distance
        dist = jnp.linalg.norm(
            cur_test_image[:, jnp.newaxis, :] - centroids[jnp.newaxis, :, :], axis=2
        )  # (num_points, num_centroid)
        cluster_assignment = jnp.argmin(dist, axis=1)[:, jnp.newaxis]  # (num_points, 1)
        no_sky_centroid_mark_broadcasted = jnp.full_like(
            cluster_assignment, no_sky_centroid_mark
        )
        masked_image = jnp.where(
            cluster_assignment < no_sky_centroid_mark_broadcasted,
            mask_color,
            cur_test_image,
        )
        result_image.append(masked_image.reshape(cur_height, cur_width, 3))

    return result_image

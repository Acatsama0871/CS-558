import jax
import itertools
import jax.numpy as jnp
from typing import Tuple
from tqdm.auto import tqdm
from .scatter_mean import scatter_mean_func


@jax.vmap
@jax.jit
def calculate_gradient(sub_image: jnp.ndarray) -> float:
    # assume sub image is 3 x 3 x channels
    return (
        jnp.linalg.norm(sub_image[2, 1] - sub_image[0, 1]) ** 2
        + jnp.linalg.norm(sub_image[1, 2] - sub_image[1, 0]) ** 2
    )


@jax.jit
def _extract_patches(sub_image: jnp.ndarray) -> jnp.ndarray:
    # extract 3 x 3 patches from 5 x 5 sub image
    patches = [
        sub_image[i - 1 : i + 2, j - 1 : j + 2]
        for i, j in itertools.product(
            range(1, sub_image.shape[0] - 1), range(1, sub_image.shape[1] - 1)
        )
    ]
    return jnp.array(patches)


def local_shift(sub_image: jnp.ndarray) -> Tuple[int, int]:
    # get gradient in 3x3
    patches = _extract_patches(sub_image)
    gradients = calculate_gradient(patches)
    lowest_gradient_index = jnp.argmin(gradients).item()

    # get local shift
    row = lowest_gradient_index // 3
    col = lowest_gradient_index % 3
    return row - 1, col - 1


def get_initial_centroid(image: jnp.ndarray, sample_freq: int) -> jnp.ndarray:
    # image: (height x width x channel)
    centroid_list = []
    for cur_x in range(sample_freq, image.shape[0], sample_freq):
        for cur_y in range(sample_freq, image.shape[1], sample_freq):
            local_shift_x, local_shift_y = local_shift(
                image[cur_x - 2 : cur_x + 3, cur_y - 2 : cur_y + 3]
            )
            cur_x += local_shift_x
            cur_y += local_shift_y
            centroid_list.append(
                [
                    cur_x,
                    cur_y,
                    image[cur_x, cur_y, 0].item(),
                    image[cur_x, cur_y, 1].item(),
                    image[cur_x, cur_y, 2].item(),
                ]
            )  # x, y, R, G, B
    return jnp.array(centroid_list).astype(jnp.float32)  # (num_centroids x 5)


def get_image_feature_vector(image: jnp.ndarray) -> jnp.ndarray:
    # image (height x width x channel) -> (height x width x 5)[x, y, R, G, B]
    x, y, _ = image.shape
    xv, yv = jnp.meshgrid(jnp.arange(x), jnp.arange(y), indexing="ij")
    coord_features = jnp.stack([xv, yv], axis=-1).reshape(-1, 2)
    color_features = image.reshape(-1, 3)
    return jnp.concatenate([coord_features, color_features], axis=-1).astype(
        jnp.float32
    )


@jax.jit
def distance_metric(
    x: jnp.ndarray, centroids: jnp.ndarray, s: float, m: float, max_dist: float
) -> jnp.ndarray:
    # x: (height x width, 1, 5)
    # centroids: (1, num_centroids, 5)
    x_rgb = x[:, :, 2:]
    x_coord = x[:, :, :2]
    centroids_rgb = centroids[:, :, 2:]
    centroids_coord = centroids[:, :, :2]

    spatial_dist = jnp.linalg.norm(x_coord - centroids_coord, axis=-1)
    mask = spatial_dist <= max_dist
    color_dist = jnp.linalg.norm(x_rgb - centroids_rgb, axis=-1)
    dist = m / s * spatial_dist + color_dist  # (num_points, num_centroids)

    return jnp.where(mask, dist, jnp.inf)


@jax.jit
def get_cluster_assignment(
    x: jnp.ndarray, centroids: jnp.ndarray, s: float, m: float, max_dist: float
) -> jnp.ndarray:
    dist = distance_metric(x=x, centroids=centroids, s=s, m=m, max_dist=max_dist)
    return jnp.argmin(dist, axis=1)


def slic_algo(
    image: jnp.ndarray, sample_freq: int, max_iter: int, tol: float, m: float
):
    # initialization
    centroids = get_initial_centroid(image, sample_freq)  # (num_centroids, 5)
    image_feature_vector = get_image_feature_vector(image)  # (height x width, 5)
    s = jnp.sqrt(image_feature_vector.shape[0] / centroids.shape[0]).item()
    centroid_dist_diff = jnp.inf

    for _ in tqdm(range(max_iter)):
        cluster_assignment = get_cluster_assignment(
            x=image_feature_vector[:, jnp.newaxis, :],
            centroids=centroids[jnp.newaxis, :, :],
            s=s,
            m=m,
            max_dist=2 * sample_freq,
        )
        prev_centroids = centroids
        centroids = scatter_mean_func(
            src=image_feature_vector,
            index=cluster_assignment,
            num_classes=centroids.shape[0],
        )
        centroid_dist_diff = jnp.linalg.norm(centroids - prev_centroids, axis=1).sum()
        if centroid_dist_diff < tol:
            break

    return centroids, cluster_assignment  # type: ignore

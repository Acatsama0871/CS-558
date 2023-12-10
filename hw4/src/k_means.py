import jax
import jax.numpy as jnp
from typing import Tuple
from tqdm.auto import tqdm


# find cluster function
@jax.jit
def find_cluster(x: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    """
    Finds the closest cluster for each data point

    Args:
        x (jnp.ndarray): data points(num_points x 1 x feature_dim)
        centroids (jnp.ndarray): centroids (1 x num_clusters x feature_dim)

    Returns:
        jnp.ndarray: cluster assignments(num_points, ) (an array of integers that indicate the cluster index for each data point)
    """
    dist = jnp.linalg.norm(
        x - centroids, axis=2
    )  # (num_cluster x feature_dim) by default this is Frobenius norm
    return jnp.argmin(dist, axis=1)


# scatter mean
@jax.jit
def scatter_add(sums, src, index):
    return sums.at[index].add(src)


@jax.jit
def scatter_count(counts, index):
    return counts.at[index].add(1)


def scatter_mean(src, index, num_classes):
    # Initialize arrays to store sums and counts
    sums = jnp.zeros((num_classes, src.shape[1]))
    counts = jnp.zeros(num_classes)

    # Accumulate sums and counts using JIT-compiled functions
    sums = scatter_add(sums, src, index)
    counts = scatter_count(counts, index)

    # Avoid division by zero
    counts = jnp.where(counts == 0, 1, counts)
    counts = counts[:, jnp.newaxis]  # type: ignore

    return sums / counts


scatter_mean = jax.jit(scatter_mean, static_argnums=(2,))


# knn fit
def kmeans_fit(
    x: jnp.ndarray,  # (num_points x feature_dim)
    num_clusters: int,
    random_seed: int,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # random initialization
    centroid_seed = jax.random.key(random_seed)
    centroids = x[
        jax.random.randint(
            key=centroid_seed, minval=0, maxval=x.shape[0], shape=(num_clusters,)
        )
    ]  # (num_clusters x feature_dim)
    prev_centroids = centroids + 1
    centroid_assignments = jnp.zeros(x.shape[0], dtype=jnp.int32)
    counter = 0
    p_bar = tqdm(total=max_iter)

    with p_bar:
        while (jnp.linalg.norm(centroids - prev_centroids, axis=1).sum() > tol) and (
            counter < max_iter
        ):
            # get centroid assignments
            centroid_assignments = find_cluster(
                x=x[:, jnp.newaxis, :], centroids=centroids[jnp.newaxis, :, :]
            )  # (num_points, ) cluster assignments for each data point

            # update centroids
            prev_centroids = centroids
            centroids = scatter_mean(
                src=x, index=centroid_assignments, num_classes=num_clusters
            )

            # update counter
            counter += 1
            p_bar.update(1)

    if prev_centroids is None:
        raise ValueError("prev_centroids is None")

    return prev_centroids, centroid_assignments

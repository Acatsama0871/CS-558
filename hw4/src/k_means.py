import jax
import jax.numpy as jnp
from typing import Tuple
from tqdm.auto import tqdm
from .scatter_mean import scatter_mean_func


# find cluster function
@jax.jit
def find_cluster(x: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    dist = jnp.linalg.norm(
        x - centroids, axis=2
    )  # (num_points, num_features) by default this is Frobenius norm
    return jnp.argmin(dist, axis=1)


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
            centroids = scatter_mean_func(
                src=x, index=centroid_assignments, num_classes=num_clusters
            )

            # update counter
            counter += 1
            p_bar.update(1)

    if prev_centroids is None:
        raise ValueError("prev_centroids is None")

    return prev_centroids, centroid_assignments

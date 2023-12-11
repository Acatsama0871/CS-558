import jax
import jax.numpy as jnp


@jax.jit
def scatter_add(sums, src, index):
    return sums.at[index].add(src)


@jax.jit
def scatter_count(counts, index):
    return counts.at[index].add(1)


def scatter_mean_func(src, index, num_classes):
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


scatter_mean_func = jax.jit(scatter_mean_func, static_argnums=(2,))

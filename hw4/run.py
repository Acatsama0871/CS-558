import os
import cv2
import typer
import numpy as np
import jax.numpy as jnp
from rich import print
from src import kmeans_fit


app = typer.Typer()


@app.command(
    "k-means-segmentation",
    help="Perform k-means segmentation on an image",
)
def k_means_segmentation_func(
    input_image_path: str = typer.Option(
        os.path.join("data", "white-tower.png"),
        "-i",
        "-input-image-path",
        help="Path to input image",
    ),
    image_output_path: str = typer.Option(
        os.path.join("result", "k_means_segmentation.png"),
        "-o",
        "-image-output-path",
        help="Path to output image",
    ),
    num_clusters: int = typer.Option(
        10, "-n", "-num-clusters", help="Number of clusters for k-means"
    ),
    random_seed: int = typer.Option(
        0, "-r", "-random-seed", help="Random seed for k-means"
    ),
    tol: float = typer.Option(
        1e-15, "-t", "-tol", help="Tolerance for k-means convergence"
    ),
    max_iter: int = typer.Option(
        2000, "-m", "-max-iter", help="Maximum number of iterations for k-means"
    ),
):
    # load image
    image = cv2.cvtColor(
        cv2.imread(input_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )

    # k-means segmentation
    image_reshaped = jnp.array(image.reshape(-1, 3))
    centroids, centroid_assignments = kmeans_fit(
        x=image_reshaped,
        num_clusters=num_clusters,
        random_seed=random_seed,
        tol=tol,
        max_iter=max_iter,
    )
    centroids = np.array(centroids.round().astype(jnp.uint8))
    centroid_assignments = np.array(centroid_assignments.reshape(image.shape[:2]))

    # save image
    image_array = np.array(
        [centroids[idx] for row in centroid_assignments for idx in row]
    )
    image_array = image_array.reshape(image.shape)
    cv2.imwrite(
        image_output_path, cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGB2BGR)
    )


if __name__ == "__main__":
    app()

import os
import cv2
import typer
import numpy as np
import jax.numpy as jnp
from src import kmeans_fit, slic_algo, sky_classification


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
        os.path.join("result"),
        "-o",
        "-image-output-path",
        help="Path to output image",
    ),
    num_clusters: int = typer.Option(
        10, "-n", "-num-clusters", help="Number of clusters for k-means"
    ),
    random_seed: int = typer.Option(
        66, "-r", "-random-seed", help="Random seed for k-means"
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
        os.path.join(
            image_output_path,
            f"kmeans_{input_image_path.split('.')[0].split('/')[1]}.png",
        ),
        cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGB2BGR),
    )


@app.command(
    "slic",
    help="Perform SLIC on an image",
)
def slic_func(
    input_image_path: str = typer.Option(
        os.path.join("data", "wt_slic.png"),
        "-i",
        "-input-image-path",
        help="Path to input image",
    ),
    image_output_path: str = typer.Option(
        os.path.join("result"),
        "-o",
        "-image-output-path",
        help="Path to output image",
    ),
    initial_point_sample_freq: int = typer.Option(
        50, "-f", "-initial-point-sample-freq", help="Initial point sample frequency"
    ),
    tol: float = typer.Option(
        1e-15, "-t", "-tol", help="Tolerance for slic convergence"
    ),
    m: float = typer.Option(
        10, "-m", "-m", help="Weight for spatial distance in distance metric"
    ),
    max_iter: int = typer.Option(
        10, "-max", "-max-iter", help="Maximum number of iterations"
    ),
):
    # load image
    image = jnp.array(
        cv2.cvtColor(cv2.imread(input_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    )
    centroids, centroid_assignments = slic_algo(
        image=image,
        sample_freq=initial_point_sample_freq,
        max_iter=max_iter,
        tol=tol,
        m=m,
    )
    centroids = centroids[:, 2:].round().astype(jnp.uint8)  # (num_centroids, r, g, b)
    centroid_assignments = centroid_assignments.reshape(
        image.shape[:2]
    )  # (height, width)

    # find the boundary
    centroid_assignment_shift_right = np.roll(centroid_assignments, 1, axis=1)
    centroid_assignment_shift_down = np.roll(centroid_assignments, 1, axis=0)
    boundary_mask = jnp.logical_or(
        centroid_assignments != centroid_assignment_shift_right,
        centroid_assignments != centroid_assignment_shift_down,
    )
    boundary_mask.at[0, :].set(False)
    boundary_mask.at[:, 0].set(False)
    image_array = centroids[centroid_assignments]
    image_array = jnp.where(boundary_mask[:, :, None], 0, image_array)
    image_array = np.array(image_array)

    cv2.imwrite(
        os.path.join(
            image_output_path,
            f"slic_{input_image_path.split('.')[0].split('/')[1]}.png",
        ),
        cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGB2BGR),
    )


@app.command(
    "sky-classification",
    help="Perform sky classification on an image",
)
def sky_classification_func(
    train_image_path: str = typer.Option(
        os.path.join("data", "sky", "train_sky.jpg"),
        "-t",
        "-train-image-path",
        help="Path to train image",
    ),
    train_masked_image_path: str = typer.Option(
        os.path.join("data", "sky", "train_sky_masked.jpg"),
        "-tm",
        "-train-masked-image-path",
        help="Path to train masked image",
    ),
    test_image_root_path: str = typer.Option(
        os.path.join("data", "sky"),
        "-e",
        "-test-image-path",
        help="Path to test image",
    ),
    image_output_path: str = typer.Option(
        os.path.join("result", "sky_output"),
        "-o",
        "-image-output-path",
        help="Path to output image",
    ),
    num_clusters: int = typer.Option(
        10, "-n", "-num-clusters", help="Number of clusters for k-means"
    ),
    random_seed: int = typer.Option(
        66, "-r", "-random-seed", help="Random seed for k-means"
    ),
    tol: float = typer.Option(
        1e-15, "-t", "-tol", help="Tolerance for k-means convergence"
    ),
    max_iter: int = typer.Option(
        2000, "-m", "-max-iter", help="Maximum number of iterations for k-means"
    ),
    mask_color_r: int = typer.Option(
        205, "-mr", "-mask-color-r", help="Mask color R value"
    ),
    mask_color_g: int = typer.Option(
        69, "-mg", "-mask-color-g", help="Mask color G value"
    ),
    mask_color_b: int = typer.Option(
        143, "-mb", "-mask-color-b", help="Mask color B value"
    ),
):
    # load train image
    train_image = jnp.array(
        cv2.cvtColor(cv2.imread(train_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    )
    train_masked_image = jnp.array(
        cv2.cvtColor(
            cv2.imread(train_masked_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
    )
    test_images = []
    test_image_names = []
    for cur_file in os.listdir(test_image_root_path):
        if cur_file.endswith(".jpg") and (not cur_file.startswith("train")):
            test_images.append(
                jnp.array(
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(test_image_root_path, cur_file),
                            cv2.IMREAD_COLOR,
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
                )
            )
            test_image_names.append(cur_file)
    # run sky classification
    results = sky_classification(
        train_image=train_image,
        train_masked_image=train_masked_image,
        mask_color=jnp.array([mask_color_r, mask_color_g, mask_color_b]),
        test_images=test_images,
        num_cluster=num_clusters,
        random_seed=random_seed,
        tol=tol,
        max_iter=max_iter,
    )
    # save images
    for cur_name, cur_result in zip(test_image_names, results):
        cv2.imwrite(
            os.path.join(
                image_output_path,
                cur_name,
            ),
            cv2.cvtColor(np.array(cur_result).astype("uint8"), cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    app()

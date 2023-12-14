import os
import cv2
import jax
import pickle
import typer
import jax.numpy as jnp
import numpy as np
from rich import print
from src import harris_detector, feature_matching_func, ransac_homography
from PIL import Image, ImageOps


app = typer.Typer()


@app.command("harris-corners", help="Harris Corners Detection")
def harris_corners_cli(
    input_image_root_path: str = typer.Option(
        os.path.join("data"),
        "-i",
        "--input-image-root-path",
        help="Input image root path",
    ),
    output_image_save_path: str = typer.Option(
        os.path.join("results", "harris_raw"),
        "-o",
        "--output-image-save-path",
        help="Output image save path",
    ),
    num_keypoints: int = typer.Option(
        1000,
        "-n",
        "--num-keypoints",
        help="Number of keypoints",
    ),
    gaussian_filter_size: int = typer.Option(
        3,
        "-g",
        "--gaussian-filter-size",
        help="Gaussian filter size",
    ),
    alpha: float = typer.Option(
        0.04,
        "-a",
        "--alpha",
        help="Alpha in Harris detector",
    ),
    nms_size: int = typer.Option(
        3,
        "-s",
        "--nms-size",
        help="Non-maximum suppression size",
    ),
    no_nms: bool = typer.Option(
        False,
        "--no-nms",
        help="Do not apply non-maximum suppression",
    ),
    sigma: float = typer.Option(
        1,
        "--sigma",
        help="Sigma in Gaussian filter",
    ),
):
    for cur_file in os.listdir(input_image_root_path):
        print(f"[blue]Processing {cur_file}...[/blue]")
        cur_image = Image.open(os.path.join(input_image_root_path, cur_file))
        cur_image = np.array(cur_image.convert("L")).astype(np.float32)
        corner_response = harris_detector(
            image=cur_image,
            alpha=alpha,
            num_keypoints=num_keypoints,
            apply_non_max_suppression=not no_nms,
            gaussian_filter_size=gaussian_filter_size,
            non_max_suppression_size=nms_size,
            sigma=sigma,
        )
        pre_fix = "harris_raw" if no_nms else "harris_nms"
        img = Image.fromarray(corner_response)
        img.convert("L").save(
            os.path.join(
                output_image_save_path, f'{pre_fix}_{cur_file.split(".")[0]}.png'
            )
        )
        print(f"[yellow]Num keypoint detected: {np.sum(corner_response > 0)}[/yellow]")


@app.command("feature-matching", help="Feature Matching")
def feature_matching_cli(
    image1_feature_path: str = typer.Option(
        os.path.join("results", "harris_nms", "harris_nms_uttower_left.png"),
        "-i1f",
        "--image1_feature",
        help="Image 1 path",
    ),
    image1_path: str = typer.Option(
        os.path.join("data", "uttower_left.jpg"),
        "-i1",
        "--image1",
        help="Image 1 path",
    ),
    image2_feature_path: str = typer.Option(
        os.path.join("results", "harris_nms", "harris_nms_uttower_right.png"),
        "-i2f",
        "--image2_feature",
        help="Image 2 path",
    ),
    image2_path: str = typer.Option(
        os.path.join("data", "uttower_right.jpg"),
        "-i2",
        "--image2",
        help="Image 2 path",
    ),
    result_save_path: str = typer.Option(
        os.path.join("results"),
        "-o",
        "--result-save-path",
        help="Result save path",
    ),
    descriptor_size: int = typer.Option(
        100, "-d", "--descriptor-size", help="Descriptor size"
    ),
    similarity_measure: str = typer.Option(
        "nss", "-s", "--similarity", help="Similarity measure"
    ),
    return_size: int = typer.Option(20, "-r", "--return-size", help="Return size"),
    return_all: bool = typer.Option(
        False,
        "--return-all",
        help="Return all keypoints",
    ),
    save_pickle: bool = typer.Option(
        False,
        "--save-pickle",
        help="Save pickle file",
    ),
):
    # load image
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # load feature
    img1_feature = Image.open(image1_feature_path)
    img1_feature = ImageOps.equalize(img1_feature)
    img1_feature = np.array(img1_feature.convert("L")).astype(np.float32)
    img2_feature = Image.open(image2_feature_path)
    img2_feature = ImageOps.equalize(img2_feature)
    img2_feature = np.array(img2_feature.convert("L")).astype(np.float32)

    # matching
    img1_keypoints, img2_keypoints, _ = feature_matching_func(
        response_image_1=img1_feature,
        response_image_2=img2_feature,
        descriptor_size=descriptor_size,
        similarity_measure=similarity_measure,
        top_k=return_size,
        return_all=return_all,
    )

    if save_pickle:
        with open(
            os.path.join(
                result_save_path,
                f"img1_{similarity_measure}_{descriptor_size}_keypoints.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump((img1_keypoints), f)
        with open(
            os.path.join(
                result_save_path,
                f"img2_{similarity_measure}_{descriptor_size}_keypoints.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump((img2_keypoints), f)
    else:
        # plot result
        concat_image = np.concatenate((img1, img2), axis=1)
        for pt1, pt2 in zip(img1_keypoints, img2_keypoints):
            cv2.circle(concat_image, (pt1[1], pt1[0]), 5, (0, 255, 0), -1)
            cv2.circle(
                concat_image, (pt2[1] + img1.shape[1], pt2[0]), 5, (0, 255, 0), -1
            )
            cv2.line(
                concat_image,
                (pt1[1], pt1[0]),
                (pt2[1] + img1.shape[1], pt2[0]),
                (0, 0, 255),
                1,
            )
        cv2.imwrite(
            os.path.join(
                result_save_path, f"concat_{similarity_measure}_{descriptor_size}.png"
            ),
            concat_image,
        )


@app.command("generate_correspondence", help="Generate Correspondence")
def generate_correspondence_cli(
    img1_correspondence_path: str = typer.Option(
        os.path.join("results", "img1_nss_160_keypoints.pkl"),
        "-i1c",
        "--img1_correspondence_path",
        help="Image 1 correspondence path",
    ),
    img2_correspondence_path: str = typer.Option(
        os.path.join("results", "img2_nss_160_keypoints.pkl"),
        "-i2c",
        "--img2_correspondence_path",
        help="Image 2 correspondence path",
    ),
    num_top_features: int = typer.Option(
        20,
        "-n",
        "--num-top-features",
        help="Number of top features",
    ),
    num_random_features: int = typer.Option(
        30,
        "-r",
        "--num-random-features",
        help="Number of random features",
    ),
    result_save_path: str = typer.Option(
        os.path.join("results"),
        "-o",
        "--result-save-path",
        help="Result save path",
    ),
    random_seed: int = typer.Option(
        0,
        "--random-seed",
        help="Random seed",
    ),
):
    with open(img1_correspondence_path, "rb") as f:
        img1_keypoints = pickle.load(f)
    with open(img2_correspondence_path, "rb") as f:
        img2_keypoints = pickle.load(f)
    # top-k
    img1_keypoints_top_k = img1_keypoints[:num_top_features]
    img2_keypoints_top_k = img2_keypoints[:num_top_features]
    img1_keypoints_top_k = jnp.array(img1_keypoints_top_k).T
    img2_keypoints_top_k = jnp.array(img2_keypoints_top_k).T
    # random
    key = jax.random.PRNGKey(random_seed)
    cur_shape = 0
    while cur_shape != num_random_features:
        random_index = np.array(
            jnp.unique(
                jax.random.randint(
                    key,
                    (num_random_features + 1,),
                    minval=0,
                    maxval=len(img1_keypoints),
                )
            )
        )
        cur_shape = random_index.shape[0]
    img1_keypoints_random = jnp.array(img1_keypoints).at[random_index].get()  # type: ignore
    img2_keypoints_random = jnp.array(img2_keypoints).at[random_index].get()  # type: ignore
    print(f"[yellow]Top-k keypoints: {img1_keypoints_top_k.shape}[/yellow]")
    print(f"[yellow]Random keypoints: {img1_keypoints_random.shape}[/yellow]")
    # save results
    jnp.save(
        os.path.join(result_save_path, "img1_keypoints_top_k.npy"), img1_keypoints_top_k
    )
    jnp.save(
        os.path.join(result_save_path, "img2_keypoints_top_k.npy"), img2_keypoints_top_k
    )
    jnp.save(
        os.path.join(result_save_path, "img1_keypoints_random.npy"),
        img1_keypoints_random,
    )
    jnp.save(
        os.path.join(result_save_path, "img2_keypoints_random.npy"),
        img2_keypoints_random,
    )


@app.command("ransac", help="RANSAC")
def ransac_cli(
    img1_path: str = typer.Option(
        os.path.join("data", "uttower_left.jpg"),
        "-i1",
        "--img1_path",
        help="Image 1 path",
    ),
    img2_path: str = typer.Option(
        os.path.join("data", "uttower_right.jpg"),
        "-i2",
        "--img2_path",
        help="Image 2 path",
    ),
    img1_correspondence_path: str = typer.Option(
        os.path.join("results", "img1_keypoints_top_k.npy"),
        "-i1c",
        "--img1_correspondence_path",
        help="Image 1 correspondence path",
    ),
    img2_correspondence_path: str = typer.Option(
        os.path.join("results", "img2_keypoints_top_k.npy"),
        "-i2c",
        "--img2_correspondence_path",
        help="Image 2 correspondence path",
    ),
    result_save_path: str = typer.Option(
        os.path.join("results", "ransac_top_k.png"),
        "-o",
        "--result-save-path",
        help="Result save path",
    ),
    dist_threshold: float = typer.Option(
        60,
        "-d",
        "--dist-threshold",
        help="Distance threshold",
    ),
    max_iter: int = typer.Option(
        1000,
        "-m",
        "--max-iter",
        help="Maximum iteration",
    ),
    sample_per_iter: int = typer.Option(
        7,
        "-p",
        "--sample-per-iter",
        help="Sample per iteration",
    ),
    random_seed: int = typer.Option(
        4,
        "--random-seed",
        help="Random seed",
    ),
):
    img1_keypoints = jnp.load(img1_correspondence_path)
    img2_keypoints = jnp.load(img2_correspondence_path)
    # ransac
    homography_matrix, inliers = ransac_homography(
        img1_keypoints=img1_keypoints,
        img2_keypoints=img2_keypoints,
        dist_threshold=dist_threshold,
        max_iter=max_iter,
        sample_per_iter=sample_per_iter,
        random_seed=random_seed,
    )


    # stitching
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    # Warp the first image
    transformed_image = cv2.warpAffine(
        image1, np.array(homography_matrix), (image1.shape[1], image1.shape[0])
    )
    result_image = cv2.addWeighted(transformed_image, 0.5, image2, 0.5, 0)
    cv2.imwrite(result_save_path, result_image)


if __name__ == "__main__":
    app()

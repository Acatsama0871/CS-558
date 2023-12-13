import os
import cv2
import pickle
import typer
import numpy as np
import jax.numpy as jnp
from rich import print
from src import harris_corner_detector, feature_similarity


app = typer.Typer()


# Problem 1
# question 1
@app.command("harris", help="Run Harris Corner Detection")
def harris_cli(
    input_root_path: str = typer.Option(
        os.path.join("data"), "-i", "--input", help="Input root path"
    ),
    output_root_path: str = typer.Option(
        os.path.join("results", "harris_raw"), "-o", "--output", help="Output root path"
    ),
    num_keypoints: int = typer.Option(
        1000, "-n", "--num-keypoints", help="Number of keypoints to show"
    ),
    gaussian_sigma: float = typer.Option(
        5,
        "-s",
        "--sigma",
        help="Sigma of Gaussian, the kernel size will be int(floor(6 * sigma + 1))",
    ),
    harris_response_alpha: float = typer.Option(
        0.05, "-a", "--alpha", help="Alpha of Harris Corner Response"
    ),
):
    for cur_file in os.listdir(input_root_path):
        print(f"[blue]Processing {cur_file}...[blue]")
        cur_img = jnp.array(
            cv2.imread(os.path.join(input_root_path, cur_file), cv2.IMREAD_GRAYSCALE)
        )
        plot_img = harris_corner_detector(
            input_img=cur_img,
            num_keypoints=num_keypoints,
            gaussian_sigma=gaussian_sigma,
            harris_response_alpha=harris_response_alpha,
            return_raw_feature_map=False,
        )
        cv2.imwrite(
            os.path.join(output_root_path, f"harris_raw_{cur_file.split('.')[0]}.png"),
            np.array(plot_img),
        )


# question 2
@app.command("harris_nms", help="Run Non-Maximum Suppression")
def non_maximum_suppression_cli(
    input_root_path: str = typer.Option(
        os.path.join("results", "harris_raw"), "-i", "--input", help="Input root path"
    ),
    output_root_path: str = typer.Option(
        os.path.join("results", "harris_nms"), "-o", "--output", help="Output root path"
    ),
    window_size: int = typer.Option(
        20, "-w", "--window-size", help="Window size of non-maximum suppression"
    ),
    num_keypoints: int = typer.Option(
        1000, "-n", "--num-keypoints", help="Number of keypoints to show"
    ),
    gaussian_sigma: float = typer.Option(
        1,
        "-s",
        "--sigma",
        help="Sigma of Gaussian, the kernel size will be int(floor(6 * sigma + 1))",
    ),
    harris_response_alpha: float = typer.Option(
        0.05, "-a", "--alpha", help="Alpha of Harris Corner Response"
    ),
):
    for cur_file in os.listdir(input_root_path):
        print(f"[blue]Processing {cur_file}...[blue]")
        cur_img = cv2.imread(os.path.join(input_root_path, cur_file))
        cur_img = cv2.GaussianBlur(cur_img, (11, 11), 0)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        cur_img = jnp.array(cur_img)
        feature_img, output_img = harris_corner_detector(
            input_img=cur_img,
            num_keypoints=num_keypoints,
            gaussian_sigma=gaussian_sigma,
            harris_response_alpha=harris_response_alpha,
            nms_window_size=window_size,
            return_raw_feature_map=True,
        )
        cv2.imwrite(
            os.path.join(
                output_root_path,
                f"harris_nms_{cur_file.split('.')[0].replace('harris_raw_', '')}.png",
            ),
            np.array(output_img.astype(jnp.uint8)),
        )
        with open(
            os.path.join(
                output_root_path,
                f"harris_nms_{cur_file.split('.')[0].replace('harris_raw_', '')}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(feature_img, f)


# question 3
@app.command("feature_match", help="Run Feature Matching")
def feature_match_cli(
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
    descriptor_size: int = typer.Option(
        10, "-d", "--descriptor-size", help="Descriptor size"
    ),
    similarity_measure: str = typer.Option(
        "ncc", "-s", "--similarity", help="Similarity measure"
    ),
    return_size: int = typer.Option(5, "-r", "--return-size", help="Return size"),
):
    # find matched keypoints
    # image1_feature = jnp.array(cv2.imread(image1_feature_path, cv2.IMREAD_GRAYSCALE))
    # image2_feature = jnp.array(cv2.imread(image2_feature_path, cv2.IMREAD_GRAYSCALE))
    with open(
        "/home/haohang/CS-558/hw3/results/harris_nms/harris_nms_uttower_left.pkl", "rb"
    ) as f:
        image1_feature = pickle.load(f)
    with open(
        "/home/haohang/CS-558/hw3/results/harris_nms/harris_nms_uttower_right.pkl", "rb"
    ) as f:
        image2_feature = pickle.load(f)
    matched_keypoints1, matched_keypoints2 = feature_similarity.match_features(
        image1_feature,
        image2_feature,
        descriptor_size=descriptor_size,
        similarity=similarity_measure,
        return_size=return_size,
    )

    # draw matched keypoints
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    combined_image = np.hstack((image1, image2))

    for pt1, pt2 in zip(matched_keypoints1, matched_keypoints2):
        pt1 = (int(pt1[1]), int(pt1[0]))  # type: ignore
        pt2 = (int(pt2[1] + image1.shape[1]), int(pt2[0]))  # type: ignore
        cv2.line(combined_image, pt1, pt2, (0, 255, 0), 1)

    cv2.imwrite(
        "test.png",
        combined_image,
    )


if __name__ == "__main__":
    app()

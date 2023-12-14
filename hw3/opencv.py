import cv2
import numpy as np


def load_images(image_path1, image_path2):
    image1 = cv2.imread("data/uttower_left.jpg")
    image2 = cv2.imread("data/uttower_left.jpg")
    return image1, image2


def detect_and_match_features(image1, image2):
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    return points1, points2


def estimate_transformation(points1, points2):
    affine_matrix, _ = cv2.estimateAffinePartial2D(points1, points2)
    return affine_matrix


def warp_image(image, affine_matrix):
    transformed_image = cv2.warpAffine(
        image, affine_matrix, (image.shape[1], image.shape[0])
    )
    return transformed_image


def stitch_images(image1, image2):
    height, width = image2.shape[:2]
    canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)

    canvas[:height, :width] = image1
    canvas[:height, width : width * 2] = image2

    return canvas


def main():
    image1, image2 = load_images("path_to_first_image.jpg", "path_to_second_image.jpg")

    points1, points2 = detect_and_match_features(image1, image2)

    affine_matrix = estimate_transformation(points1, points2)

    transformed_image = warp_image(image1, affine_matrix)

    stitched_image = stitch_images(transformed_image, image2)

    cv2.imwrite("stitched_image.jpg", stitched_image)


if __name__ == "__main__":
    main()

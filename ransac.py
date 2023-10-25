import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm


def random_n_points_pair(points, n):
    idx = np.random.choice(points.shape[0], n, replace=False)
    return np.array(list(itertools.combinations(points[idx], 2)))


def fit_line(points):
    m = np.vstack(points)
    m = np.hstack([m, np.ones((m.shape[0], 1))])
    eigen_values, eigen_vectors = np.linalg.eig(m.T @ m)
    index = np.argmin(eigen_values)
    a, b, c = eigen_vectors[:, index]
    norm = (a**2 + b**2)**0.5
    a, b, c = a / norm, b / norm, c / norm
    return a, b, c

def distance_to_line(p, a, b, c):
    x, y = p
    return abs(a * x + b * y + c)

def ransac_one_iter(all_points, num_point_pairs=1000, threshold=2):
    point_pairs = random_n_points_pair(all_points, num_point_pairs)
    
    # find the line with most inliers
    best_inlier_count = 0
    best_line_inliers = None
    for cur_pair in tqdm(point_pairs):
        a, b, c = fit_line([cur_pair[0], cur_pair[1]])
        cur_inliers = [p for p in all_points if distance_to_line(p, a, b, c) < threshold]
        inlier_count = len(cur_inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_line_inliers = cur_inliers
    
    # refit the line with all inliers
    a, b, c = fit_line(best_line_inliers)
    all_points = np.array([p for p in all_points if distance_to_line(p, a, b, c) >= threshold])
    return a, b, c, all_points, len(best_line_inliers), best_line_inliers

def ransac(all_points, num_lines, num_point_pairs=300, threshold=2):
    all_lines = []
    for _ in range(num_lines):
        a, b, c, all_points, num_inliers, inliers = ransac_one_iter(all_points, num_point_pairs, threshold)
        all_lines.append([a, b, c, num_inliers, inliers])
    return all_lines
    
def get_y_from_x(x, a, b, c):
    return -(a * x + c) / b

if __name__ == "__main__":
    # load image and keypoints
    image = cv2.imread('/home/haohang/CS-558/hw2/result/keypoint.png', 0)
    y, x = np.where(image > 0)
    x, y = x.astype(np.float32), y.astype(np.float32)
    
    # ransac
    all_points = np.stack([x, y], axis=1)
    result = ransac(all_points, 4, 300, 2)
   
    # plot
    plt.imshow(image, cmap='gray')
    for cur_r in result:
        a, b, c, _, inliers = cur_r
        x, y = np.stack(inliers, axis=1)
        line_x = np.arange(x.min(), x.max(), 0.1)
        line_y = get_y_from_x(line_x, a, b, c)
        plt.scatter(x, y, color='yellow')
        plt.plot(line_x, line_y, color='blue', linewidth=2)
    plt.savefig('ransac.png')
    

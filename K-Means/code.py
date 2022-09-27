import numpy as np
from matplotlib import pyplot as plt

RESULTS_DIR = 'results/'
POINTS_PATH = 'points.txt'
PLOT_PATH = 'res01.jpg'
OUTPUT_PATHS = ['res02.jpg', 'res03.jpg']
IMPROVED_PATH = 'res04.jpg'

CLUSTERS = 2


def read_points():
    with open(POINTS_PATH, 'r') as f:
        lines = f.readlines()
    return list(map(lambda line: (float(line.split()[0]), float(line.split()[1])), lines[1:1 + int(lines[0])]))


def save_plot(path, data, title=None, labels=None):
    plt.title(title)
    if labels is None:
        plt.scatter(data[:, 0], data[:, 1])
    else:
        colored_data = np.hstack((data, labels.reshape(-1, 1)))
        plt.scatter(colored_data[:, 0], colored_data[:, 1], c=colored_data[:, 2])
    plt.savefig(f'{RESULTS_DIR}{path}', bbox_inches='tight')


def add_location_feature(X):
    return np.hstack((X, np.linalg.norm(X, axis=1).reshape(-1, 1)))


def k_means(K, X):
    MAX_ITER = 1000

    def calc_distance(X1, X2):
        centers_count, data_size = X2.shape[0], X1.shape[0]
        X1_norm = np.repeat((np.linalg.norm(X1, axis=1) ** 2).reshape(-1, 1), centers_count, axis=1)
        X2_norm = np.repeat((np.linalg.norm(X2, axis=1) ** 2).reshape(1, -1), data_size, axis=0)

        return X1_norm + X2_norm - 2 * X1 @ X2.T

    def convergence_reached(old, new):
        return np.array(old == new).all()

    centers = np.random.randn(K, X.shape[1]) * np.mean(X, axis=0) + np.std(X, axis=0)
    for i in range(MAX_ITER):
        data_center_dists = calc_distance(X, centers)
        labels = data_center_dists.argmin(axis=1)

        old_centers = np.copy(centers)
        for k in range(K):
            cluster_k = X[labels == k]
            if len(cluster_k) > 0:
                centers[k] = cluster_k.mean(axis=0)

        if convergence_reached(old_centers, centers):
            return labels


points = np.array(read_points())
save_plot(PLOT_PATH, points, title='Points before clustering')
for t in range(2):
    result = k_means(CLUSTERS, points)
    save_plot(OUTPUT_PATHS[t], points, labels=result, title=f'K-Means Run {t + 1}')

new_points = add_location_feature(points)
result = k_means(CLUSTERS, new_points)
save_plot(IMPROVED_PATH, points, labels=result, title='K-Means clustering using additional column')

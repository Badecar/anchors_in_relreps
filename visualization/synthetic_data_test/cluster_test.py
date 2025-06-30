import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# -------------------------------
# Helper functions for dataset creation
# -------------------------------
def create_cluster(center, cov, n_points, label):
    """Generates n_points for a single cluster with given center and covariance."""
    points = np.random.multivariate_normal(mean=center, cov=cov, size=n_points)
    labels = np.full(n_points, label)
    return points, labels

def create_dataset(centers, cov_params, points_per_cluster):
    """
    Creates a synthetic dataset.
    centers: list of [x,y] where each cluster center is defined.
    cov_params: list of dicts for each cluster. Each dict should include keys: 'var_x', 'var_y', 'corr'
      where cov matrix = [[var_x, corr*sqrt(var_x*var_y)], [corr*sqrt(var_x*var_y), var_y]]
    points_per_cluster: number of points per each cluster.
    
    Returns:
      X: np.array of shape [N,2]
      y: np.array of labels
    """
    data = []
    labels = []
    for i, center in enumerate(centers):
        var_x = cov_params[i]['var_x']
        var_y = cov_params[i]['var_y']
        corr = cov_params[i]['corr']
        cov_xy = corr * np.sqrt(var_x * var_y)
        cov = [[var_x, cov_xy], [cov_xy, var_y]]
        pts, lab = create_cluster(center, cov, points_per_cluster, i)
        data.append(pts)
        labels.append(lab)
    X = np.vstack(data)
    y = np.hstack(labels)
    return X, y

# -------------------------------
# Main script configuration
# -------------------------------
if __name__ == '__main__':
    np.random.seed(42)

    # Choose metric: "cosine", "euclidean", or "mahalanobis"
    metric = "mahalanobis"  # change this to "euclidean" or "mahalanobis" as desired

    # Define clusters: number of clusters, centers and covariance parameters.
    centers = [
        [2, 2],
        [8, 3],
        [4, 5]
    ]

    # Define manual anchors (2 anchors for this example) as a list
    anchor1 = np.array([1, 1])
    anchor2 = np.array([5, 7])
    anchor3 = np.array([9, 7])
    anchors = [anchor1, anchor2]

    cov_params = [
        {'var_x': 1.0, 'var_y': 0.5, 'corr': 0.3},
        {'var_x': 0.8, 'var_y': 1.2, 'corr': -0.2},
        {'var_x': 1.5, 'var_y': 1.0, 'corr': 0.5}
    ]
    points_per_cluster = 200

    # Create dataset.
    X, y = create_dataset(centers, cov_params, points_per_cluster)

    # Scale the entire space by a given factor.
    scale_factor = 6  # adjust this value to scale the space
    X = scale_factor * X
    anchors = [scale_factor * a for a in anchors]

    # New transformation parameters
    rotate_angle_deg = 70  # Rotation angle in degrees; adjust as desired
    translation_vector = np.array([10, 6])  # Translation vector; adjust as desired

    # Convert rotation angle to radians.
    rotate_angle = np.radians(rotate_angle_deg)

    # Compute the midmost datapoint (using the median)
    center_point = np.median(X, axis=0)

    # Create a 2D rotation matrix.
    R = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],
                [np.sin(rotate_angle),  np.cos(rotate_angle)]])

    # Rotate dataset around the center_point.
    X = (X - center_point) @ R.T + center_point

    # Translate dataset by the given translation vector.
    X = X + translation_vector

    # Apply the same transformations to the anchors.
    anchors = [ (a - center_point) @ R.T + center_point for a in anchors ]
    anchors = [ a + translation_vector for a in anchors ]

    # Plot the absolute dataset with anchors overlaid
    plt.figure(figsize=(6,6))
    plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=0.6)
    for i, anchor in enumerate(anchors):
        if i == 0:
            plt.scatter(anchor[0], anchor[1], c='red', marker='X', s=150, label='Anchors')
        else:
            plt.scatter(anchor[0], anchor[1], c='red', marker='X', s=150)
    plt.axhline(0, color='black', linewidth=3)  # thick horizontal line at y=0
    plt.axvline(0, color='black', linewidth=3)  # thick vertical line at x=0

    # Set fixed coordinate limits (adjust as desired)
    plt.xlim(-5, 75)
    plt.ylim(-5, 75)

    plt.title('Absolute Data with Anchors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------
    # Compute relative representations manually.
    if metric == "cosine":
        # Compute cosine similarities manually using loops.
        rel_X = np.empty((X.shape[0], len(anchors)))
        for i, embedding in enumerate(X):
            norm_embedding = np.linalg.norm(embedding)
            emb_unit = embedding if norm_embedding == 0 else embedding / norm_embedding
            scores = []
            for a in anchors:
                norm_anchor = np.linalg.norm(a)
                anchor_unit = np.zeros_like(a) if norm_anchor == 0 else a / norm_anchor
                score = np.dot(emb_unit, anchor_unit)
                scores.append(score)
            rel_X[i, :] = scores
    elif metric == "euclidean":
        # Compute Euclidean distances manually using loops.
        rel_X = np.empty((X.shape[0], len(anchors)))
        for i, embedding in enumerate(X):
            distances = []
            for a in anchors:
                dist = np.linalg.norm(embedding - a)
                distances.append(dist)
            rel_X[i, :] = distances
    elif metric == "mahalanobis":
        # Compute Mahalanobis distances manually using loops.
        # First, compute sample covariance matrix for X and its inverse.
        cov_X = np.cov(X, rowvar=False)
        inv_cov_X = np.linalg.inv(cov_X)
        rel_X = np.empty((X.shape[0], len(anchors)))
        for i, embedding in enumerate(X):
            distances = []
            for a in anchors:
                diff = embedding - a
                dist = np.dot(np.dot(diff, inv_cov_X), diff.T)
                distances.append(dist)
            rel_X[i, :] = distances
    else:
        raise ValueError("Unknown metric provided. Use 'cosine', 'euclidean', or 'mahalanobis'.")

    # Plot the relative representations.
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap("viridis")
    unique_labels = np.unique(y)
    for label in unique_labels:
        color = cmap(label / (len(unique_labels) - 1))  # Normalize label to [0,1]
        plt.scatter(rel_X[y == label, 0], rel_X[y == label, 1],
                    label=f'Class {label}', color=color, alpha=0.6)
    plt.axhline(0, color='black', linewidth=3)  # thick horizontal line at y=0
    plt.axvline(0, color='black', linewidth=3)  # thick vertical line at x=0

    # Set fixed coordinate limits (adjust as desired)
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)

    plt.title('Relative Representation - Mahalanobis')
    plt.xlabel('Feature from Anchor 1')
    plt.ylabel('Feature from Anchor 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------
    # Train classifiers on absolute data and relative representations.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rel_X_train, rel_X_test, _, _ = train_test_split(rel_X, y, test_size=0.3, random_state=42)

    # Classifier on absolute data
    clf_abs = LogisticRegression(max_iter=500).fit(X_train, y_train)
    y_pred_abs = clf_abs.predict(X_test)
    f1_abs = f1_score(y_test, y_pred_abs, average='weighted')

    # Classifier on relative representations
    clf_rel = LogisticRegression(max_iter=500).fit(rel_X_train, y_train)
    y_pred_rel = clf_rel.predict(rel_X_test)
    f1_rel = f1_score(y_test, y_pred_rel, average='weighted')

    print("F1 score on absolute data: {:.4f}".format(f1_abs))
    print("F1 score on relative representations: {:.4f}".format(f1_rel))




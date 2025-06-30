import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def nearest_positive_semidefinite(A):
    # Ensure symmetry
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals[eigvals < 1e-8] = 1e-8  # threshold to prevent negative or zero eigenvalues
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def generate_random_covariance(dims, low=0.5, high=2.0):
    if dims == 1:
        var = np.random.uniform(low, high)
        return np.array([[var]])
    stds = np.sqrt(np.random.uniform(low, high, size=dims))
    R = np.eye(dims)
    for i in range(dims):
        for j in range(i+1, dims):
            r = np.random.uniform(-0.8, 0.8)
            R[i, j] = r
            R[j, i] = r
    cov = np.outer(stds, stds) * R
    cov = nearest_positive_semidefinite(cov)
    return cov

def create_cluster(center, cov, n_points, label):
    points = np.random.multivariate_normal(mean=center, cov=cov, size=n_points)
    labels = np.full(n_points, label)
    return points, labels

def create_dataset(num_clusters, dims, points_per_cluster):
    data = []
    labels = []
    centers = []
    for i in range(num_clusters):
        center = np.random.uniform(-3, 3, size=dims)
        centers.append(center)
        cov = generate_random_covariance(dims)
        pts, lab = create_cluster(center, cov, points_per_cluster, i)
        data.append(pts)
        labels.append(lab)
    X = np.vstack(data)
    y = np.hstack(labels)
    centers = np.array(centers)
    return X, y, centers

def compute_relative_cosine(X, anchors):
    rel_X = np.empty((X.shape[0], anchors.shape[0]))
    for i, emb in enumerate(X):
        norm_emb = np.linalg.norm(emb)
        emb_unit = emb if norm_emb == 0 else emb / norm_emb
        scores = []
        for a in anchors:
            norm_a = np.linalg.norm(a)
            a_unit = np.zeros_like(a) if norm_a == 0 else a / norm_a
            score = np.dot(emb_unit, a_unit)
            scores.append(score)
        rel_X[i, :] = scores
    return rel_X

def compute_relative_euclidean(X, anchors):
    rel_X = np.empty((X.shape[0], anchors.shape[0]))
    for i, emb in enumerate(X):
        distances = []
        for a in anchors:
            dist = np.linalg.norm(emb - a)
            distances.append(dist)
        rel_X[i, :] = distances
    return rel_X

def compute_relative_mahalanobis(X, anchors):
    cov_X = np.cov(X, rowvar=False)
    inv_cov_X = np.linalg.inv(cov_X)
    rel_X = np.empty((X.shape[0], anchors.shape[0]))
    for i, emb in enumerate(X):
        distances = []
        for a in anchors:
            diff = emb - a
            dist = np.sqrt(np.dot(np.dot(diff, inv_cov_X), diff.T))
            distances.append(dist)
        rel_X[i, :] = distances
    return rel_X

def random_transformation(X, dims, rotation=None, scale=None, translation=None):
    if rotation is None:
        A = np.random.randn(dims, dims)
        Q, _ = np.linalg.qr(A)
    else:
        Q = rotation
    if scale is None:
        scale = np.random.uniform(0.2, 5.0, size=dims)
    if translation is None:
        translation = np.array([np.random.choice([-1, 1]) * np.random.uniform(1, 3) for _ in range(dims)])
    X_trans = (X.dot(Q)) * scale + translation
    return X_trans, Q, scale, translation

def run_experiment(seed, dims, num_clusters, points_per_cluster, num_anchors, use_cluster_centers, noise_sigma, plot):
    np.random.seed(seed)
    # Create first (original) dataset.
    X1, y, cluster_centers = create_dataset(num_clusters, dims, points_per_cluster)
    # Select anchors from the first dataset.
    if use_cluster_centers:
        if num_anchors <= num_clusters:
            anchors1 = cluster_centers[:num_anchors]
        else:
            extra = num_anchors - num_clusters
            extra_anchors = np.random.uniform(-3, 3, size=(extra, dims))
            anchors1 = np.vstack((cluster_centers, extra_anchors))
    else:
        anchors1 = np.random.uniform(-3, 3, size=(num_anchors, dims))
    
    # Compute relative representations on dataset1.
    rel_cosine_1 = compute_relative_cosine(X1, anchors1)
    rel_euclidean_1 = compute_relative_euclidean(X1, anchors1)
    rel_mahalanobis_1 = compute_relative_mahalanobis(X1, anchors1)
    
    # Split dataset1 for training/testing.
    X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=42)
    rel_cosine_train, rel_cosine_test, _, _ = train_test_split(rel_cosine_1, y, test_size=0.3, random_state=42)
    rel_euclidean_train, rel_euclidean_test, _, _ = train_test_split(rel_euclidean_1, y, test_size=0.3, random_state=42)
    rel_mahalanobis_train, rel_mahalanobis_test, _, _ = train_test_split(rel_mahalanobis_1, y, test_size=0.3, random_state=42)
    
    # Train classifiers on the absolute and relative representations.
    clf_abs = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3).fit(X1_train, y_train)
    clf_cos = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3).fit(rel_cosine_train, y_train)
    clf_euc = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3).fit(rel_euclidean_train, y_train)
    clf_mah = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3).fit(rel_mahalanobis_train, y_train)
    
    # Evaluate on dataset1.
    y_pred_cos_1 = clf_cos.predict(rel_cosine_test)
    f1_cos_1 = f1_score(y_test, y_pred_cos_1, average='weighted')
    y_pred_euc_1 = clf_euc.predict(rel_euclidean_test)
    f1_euc_1 = f1_score(y_test, y_pred_euc_1, average='weighted')
    y_pred_mah_1 = clf_mah.predict(rel_mahalanobis_test)
    f1_mah_1 = f1_score(y_test, y_pred_mah_1, average='weighted')
    
    # Build dataset2 by applying a transformation.
    override_rotation = None
    override_scaling = None
    override_translation = None
    X2, Q, scale, translation = random_transformation(X1, dims, rotation=override_rotation,
                                                       scale=override_scaling, translation=override_translation)
    anchors2 = (anchors1.dot(Q)) * scale + translation
    if noise_sigma > 0.0:
        X2 = X2 + noise_sigma * np.random.randn(*X2.shape)
    
    # Compute relative representations on dataset2.
    rel_cosine_2 = compute_relative_cosine(X2, anchors2)
    rel_euclidean_2 = compute_relative_euclidean(X2, anchors2)
    rel_mahalanobis_2 = compute_relative_mahalanobis(X2, anchors2)
    
    # Split dataset2 similarly.
    _, X2_test, _, _ = train_test_split(X2, y, test_size=0.3, random_state=42)
    rel_cosine_2_train, rel_cosine_2_test, _, _ = train_test_split(rel_cosine_2, y, test_size=0.3, random_state=42)
    rel_euclidean_2_train, rel_euclidean_2_test, _, _ = train_test_split(rel_euclidean_2, y, test_size=0.3, random_state=42)
    rel_mahalanobis_2_train, rel_mahalanobis_2_test, _, _ = train_test_split(rel_mahalanobis_2, y, test_size=0.3, random_state=42)
    
    # Evaluate classifiers (trained on dataset1) on dataset2.
    y_pred_cos_2 = clf_cos.predict(rel_cosine_2_test)
    f1_cos_2 = f1_score(y_test, y_pred_cos_2, average='weighted')
    y_pred_euc_2 = clf_euc.predict(rel_euclidean_2_test)
    f1_euc_2 = f1_score(y_test, y_pred_euc_2, average='weighted')
    y_pred_mah_2 = clf_mah.predict(rel_mahalanobis_2_test)
    f1_mah_2 = f1_score(y_test, y_pred_mah_2, average='weighted')
    
    if plot:
        # Visualize using PCA if enabled.
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        titles_row1 = [
            "Absolute Dataset1" + (" (PCA)" if dims > 2 else ""),
            "Relative Cosine Dataset1" + (" (PCA)" if dims > 2 else ""),
            "Relative Euclidean Dataset1" + (" (PCA)" if dims > 2 else ""),
            "Relative Mahalanobis Dataset1" + (" (PCA)" if dims > 2 else "")
        ]
        titles_row2 = [
            "Absolute Dataset2" + (" (PCA)" if dims > 2 else ""),
            "Relative Cosine Dataset2" + (" (PCA)" if dims > 2 else ""),
            "Relative Euclidean Dataset2" + (" (PCA)" if dims > 2 else ""),
            "Relative Mahalanobis Dataset2" + (" (PCA)" if dims > 2 else "")
        ]
        datasets_row1 = [X1, rel_cosine_1, rel_euclidean_1, rel_mahalanobis_1]
        anchors_list_row1 = [
            anchors1,
            compute_relative_cosine(anchors1, anchors1),
            compute_relative_euclidean(anchors1, anchors1),
            compute_relative_mahalanobis(anchors1, anchors1)
        ]
        datasets_row2 = [X2, rel_cosine_2, rel_euclidean_2, rel_mahalanobis_2]
        anchors_list_row2 = [
            anchors2,
            compute_relative_cosine(anchors2, anchors2),
            compute_relative_euclidean(anchors2, anchors2),
            compute_relative_mahalanobis(anchors2, anchors2)
        ]
        for row, datasets, anchors_list, titles in zip([0, 1], [datasets_row1, datasets_row2], [anchors_list_row1, anchors_list_row2], [titles_row1, titles_row2]):
            for col in range(4):
                ax = axes[row, col]
                data = datasets[col]
                anchors_rep = anchors_list[col]
                if dims > 2:
                    pca = PCA(n_components=2)
                    data_2d = pca.fit_transform(data)
                    anchors_2d = pca.transform(anchors_rep)
                else:
                    data_2d = data
                    anchors_2d = anchors_rep
                sc = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
                ax.scatter(anchors_2d[:, 0], anchors_2d[:, 1], c='red', marker='x', s=100, label='Anchors')
                ax.set_title(titles[col])
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.grid(True)
                ax.axhline(0, color='black', lw=2)
                ax.axvline(0, color='black', lw=2)
                fig.colorbar(sc, ax=ax)
                ax.legend()
        plt.tight_layout()
        plt.show()

    return f1_cos_1, f1_cos_2, f1_euc_1, f1_euc_2, f1_mah_1, f1_mah_2

if __name__ == '__main__':
    dims = 20                      # number of dimensions
    num_clusters = dims * 3        # number of clusters
    points_per_cluster = 500       # points for each cluster
    num_anchors = dims + 1         # number of anchors to use
    use_cluster_centers = False     # whether to use cluster centers as anchors
    noise_sigma = 2                # noise level for dataset2
    plot = False                 # plotting disabled by default in loop
    n_seeds = 10                   # number of runs (seeds)

    results_cos_1 = []
    results_cos_2 = []
    results_euc_1 = []
    results_euc_2 = []
    results_mah_1 = []
    results_mah_2 = []
    
    print("Starting experiments...\n")
    for i in range(n_seeds):
        seed = 42 + i
        f1_cos_1, f1_cos_2, f1_euc_1, f1_euc_2, f1_mah_1, f1_mah_2 = run_experiment(
            seed, dims, num_clusters, points_per_cluster, num_anchors, use_cluster_centers, noise_sigma, plot)
        results_cos_1.append(f1_cos_1)
        results_cos_2.append(f1_cos_2)
        results_euc_1.append(f1_euc_1)
        results_euc_2.append(f1_euc_2)
        results_mah_1.append(f1_mah_1)
        results_mah_2.append(f1_mah_2)
        print("Seed {}:".format(seed))
        print("  Cosine - Dataset1: {:.4f} | Dataset2: {:.4f} | Diff: {:.4f}".format(f1_cos_1, f1_cos_2, abs(f1_cos_1 - f1_cos_2)))
        print("  Euclidean - Dataset1: {:.4f} | Dataset2: {:.4f} | Diff: {:.4f}".format(f1_euc_1, f1_euc_2, abs(f1_euc_1 - f1_euc_2)))
        print("  Mahalanobis - Dataset1: {:.4f} | Dataset2: {:.4f} | Diff: {:.4f}\n".format(f1_mah_1, f1_mah_2, abs(f1_mah_1 - f1_mah_2)))

    def mean_std(scores):
        return np.mean(scores), np.std(scores)

    m_cos1, s_cos1 = mean_std(results_cos_1)
    m_cos2, s_cos2 = mean_std(results_cos_2)
    m_euc1, s_euc1 = mean_std(results_euc_1)
    m_euc2, s_euc2 = mean_std(results_euc_2)
    m_mah1, s_mah1 = mean_std(results_mah_1)
    m_mah2, s_mah2 = mean_std(results_mah_2)

    print("----- Summary over {} seeds -----".format(n_seeds))
    print("Cosine Representation:")
    print("  Dataset1: Mean = {:.4f}, Std = {:.4f}".format(m_cos1, s_cos1))
    print("  Dataset2: Mean = {:.4f}, Std = {:.4f}".format(m_cos2, s_cos2))
    print("Euclidean Representation:")
    print("  Dataset1: Mean = {:.4f}, Std = {:.4f}".format(m_euc1, s_euc1))
    print("  Dataset2: Mean = {:.4f}, Std = {:.4f}".format(m_euc2, s_euc2))
    print("Mahalanobis Representation:")
    print("  Dataset1: Mean = {:.4f}, Std = {:.4f}".format(m_mah1, s_mah1))
    print("  Dataset2: Mean = {:.4f}, Std = {:.4f}".format(m_mah2, s_mah2))
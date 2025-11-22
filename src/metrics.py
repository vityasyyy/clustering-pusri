import numpy as np


# wcss for the loss func, compute tightness of all points to their centroids
def get_wcss(particle, data, k):
    """Objective Function: Within-Cluster Sum of Squares"""
    centroids = particle.reshape(k, data.shape[1])
    distances = np.zeros((data.shape[0], k))
    for i, c in enumerate(centroids):
        distances[:, i] = np.sum((data - c) ** 2, axis=1)
    return np.sum(np.min(distances, axis=1))


# calculate all cohesion(distance to own centroid) and separation(distance to other centrooid) of the aps
def silhouette_score_scratch(data, labels):
    n = data.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0

    s_scores = []
    for i in range(n):
        point = data[i]
        label = labels[i]

        # a(i): Cohesion (Mean distance to own cluster)
        own_cluster = data[labels == label]
        if len(own_cluster) > 1:
            # Distance to all other points in own cluster
            dists = np.sqrt(np.sum((own_cluster - point) ** 2, axis=1))
            a_i = np.mean(dists[dists > 0])  # Exclude self-distance (0)
        else:
            a_i = 0

        # b(i): Separation (Mean distance to nearest neighbor cluster)
        b_i = float("inf")
        for other_l in unique_labels:
            if other_l == label:
                continue
            other_cluster = data[labels == other_l]
            if len(other_cluster) > 0:
                avg_dist = np.mean(
                    np.sqrt(np.sum((other_cluster - point) ** 2, axis=1))
                )
                if avg_dist < b_i:
                    b_i = avg_dist

        s_scores.append((b_i - a_i) / max(a_i, b_i))

    return np.mean(s_scores)

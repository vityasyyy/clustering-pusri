import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. DATA PREPROCESSING
# ==========================================
print("Loading and Preprocessing Data...")
df = pd.read_csv("average_metrics_uap_5min.csv")

# Drop rows with missing 'Core Metrics' (Client, CPU, Memory)
# Justification: We cannot optimize performance for APs that aren't reporting status.
df_clean = df.dropna(
    subset=["avg_client_count", "avg_cpu_usage_ratio", "avg_memory_usage_ratio"]
).copy()

# Impute missing 'Signal Metrics' with Noise Floor (-95 dBm)
# Justification: Missing signal usually means 'No Signal' / 'Radio Off', not bad data.
df_clean["avg_signal_5g_dbm"] = df_clean["avg_signal_5g_dbm"].fillna(-95)
df_clean["avg_signal_24g_dbm"] = df_clean["avg_signal_24g_dbm"].fillna(-95)

# Select features for clustering
feature_cols = [
    "avg_client_count",
    "avg_cpu_usage_ratio",
    "avg_memory_usage_ratio",
    "avg_signal_5g_dbm",
    "avg_signal_24g_dbm",
]

# Convert to Numpy array of Floats (to avoid type errors)
X_raw = df_clean[feature_cols].to_numpy(dtype=float)

# Standardization (Z-Score) - ESSENTIAL for distance-based algorithms
# Justification: CPU is 0-1, Signal is -90 to -30. Scaling makes them comparable.
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std

# ==========================================
# 2. ALGORITHMS FROM SCRATCH (No Sklearn Clustering)
# ==========================================


def get_wcss(particle, data, k):
    """Objective Function: Within-Cluster Sum of Squares"""
    centroids = particle.reshape(k, data.shape[1])
    distances = np.zeros((data.shape[0], k))
    for i, c in enumerate(centroids):
        distances[:, i] = np.sum((data - c) ** 2, axis=1)
    return np.sum(np.min(distances, axis=1))


def silhouette_score_scratch(data, labels):
    """Validation Metric: Silhouette Score (Separation vs Cohesion)"""
    n = data.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0

    s_scores = []
    # For efficiency on larger datasets, you might sample, but for <1000 rows, full calc is fine.
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


def run_pso(data, k, n_particles=20, max_iters=30):
    """Particle Swarm Optimization for Clustering"""
    dim = k * data.shape[1]

    # Initialize Swarm
    particles = np.zeros((n_particles, dim))
    for i in range(n_particles):
        # Initialize particles with random data points to start in a valid area
        particles[i] = data[np.random.choice(data.shape[0], k)].flatten()

    velocities = np.random.uniform(-0.1, 0.1, (n_particles, dim))

    # Personal Bests
    pbest_pos = particles.copy()
    pbest_scores = np.array([get_wcss(p, data, k) for p in particles])

    # Global Best
    gbest_idx = np.argmin(pbest_scores)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    history = [gbest_score]

    # PSO Parameters
    w = 0.729  # Inertia
    c1 = 1.494  # Cognitive (Self)
    c2 = 1.494  # Social (Swarm)

    for _ in range(max_iters):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # Update Velocity
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_pos[i] - particles[i])
                + c2 * r2 * (gbest_pos - particles[i])
            )

            # Update Position
            particles[i] += velocities[i]

            # Evaluate
            score = get_wcss(particles[i], data, k)

            # Update Personal Best
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_pos[i] = particles[i].copy()

                # Update Global Best
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = particles[i].copy()

        history.append(gbest_score)

    return gbest_pos, history


# ==========================================
# 3. OPTIMAL K ANALYSIS (Using PSO)
# ==========================================
print("\n--- Step 1: Finding Optimal k using PSO ---")
k_range = range(2, 7)
wcss_results = []
silhouette_results = []
best_overall_score = -1
best_k = -1
best_centroids_final = None

for k in k_range:
    print(f"Swarm optimizing for k={k}...")

    # 1. Run PSO
    best_centroids_flat, _ = run_pso(X, k)

    # 2. Get Labels
    centroids = best_centroids_flat.reshape(k, X.shape[1])
    dists = np.zeros((X.shape[0], k))
    for i, c in enumerate(centroids):
        dists[:, i] = np.sum((X - c) ** 2, axis=1)
    labels = np.argmin(dists, axis=1)

    # 3. Calculate Metrics
    wcss_val = get_wcss(best_centroids_flat, X, k)
    sil_val = silhouette_score_scratch(X, labels)

    wcss_results.append(wcss_val)
    silhouette_results.append(sil_val)

    if sil_val > best_overall_score:
        best_overall_score = sil_val
        best_k = k
        best_centroids_final = centroids
        best_labels_final = labels

print(
    f"\nAnalysis Complete. Best k = {best_k} (Silhouette Score: {best_overall_score:.4f})"
)

# Plot k Analysis
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel("k (Number of Clusters)")
ax1.set_ylabel("WCSS (Cost)", color="tab:blue")
ax1.plot(k_range, wcss_results, marker="o", color="tab:blue", label="WCSS (PSO)")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Silhouette Score (Quality)", color="tab:orange")
ax2.plot(
    k_range,
    silhouette_results,
    marker="s",
    linestyle="--",
    color="tab:orange",
    label="Silhouette",
)
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title("Optimal k Analysis (Using PSO Only)")
plt.grid(True)
fig.tight_layout()
plt.savefig("final_k_analysis.png")

# ==========================================
# 4. SAVE & VISUALIZE FINAL RESULTS
# ==========================================
print(f"\n--- Step 2: Finalizing Results for k={best_k} ---")

# Assign Cluster Names (Interpretation)
# We need to see the mean values of original data to name them
df_clean["Cluster"] = best_labels_final
cluster_stats = df_clean.groupby("Cluster")[feature_cols].mean()
print("\nCluster Profiles (Mean Values):")
print(cluster_stats)

# Visualize Final Clusters (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(best_centroids_final)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=best_labels_final,
    cmap="viridis",
    alpha=0.6,
    label="Data Points",
)
plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    s=200,
    c="red",
    marker="X",
    label="Centroids",
)
plt.title(f"Final PSO Clustering Results (k={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.colorbar(scatter, label="Cluster ID")
plt.savefig("final_clusters_pso.png")

# Save to CSV
output_file = "clustered_uap_metrics_pso.csv"
df_clean.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import from our new modules
from data_loader import load_and_process_data
from metrics import get_wcss, silhouette_score_scratch
from pso_clustering import run_pso

# 1. LOAD DATA
X, df_clean, feature_cols = load_and_process_data("average_metrics_uap_5min.csv")

# 2. FIND OPTIMAL K
print("\n--- Step 1: Finding Optimal k using PSO ---")
k_range = range(2, 7)
wcss_results = []
silhouette_results = []
best_overall_score = -1
best_k = -1
best_centroids_final = None
best_labels_final = None

for k in k_range:
    print(f"Swarm optimizing for k={k}...")

    # Run PSO
    best_centroids_flat, _ = run_pso(X, k)

    # Get Labels based on the best centroids found
    centroids = best_centroids_flat.reshape(k, X.shape[1])
    dists = np.zeros((X.shape[0], k))
    for i, c in enumerate(centroids):
        dists[:, i] = np.sum((X - c) ** 2, axis=1)
    labels = np.argmin(dists, axis=1)

    # Calculate Metrics
    wcss_val = get_wcss(best_centroids_flat, X, k)
    sil_val = silhouette_score_scratch(X, labels)

    wcss_results.append(wcss_val)
    silhouette_results.append(sil_val)

    # Track the best result
    if sil_val > best_overall_score:
        best_overall_score = sil_val
        best_k = k
        best_centroids_final = centroids
        best_labels_final = labels

print(
    f"\nAnalysis Complete. Best k = {best_k} (Silhouette Score: {best_overall_score:.4f})"
)

# 3. VISUALIZE K ANALYSIS
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
plt.tight_layout()
plt.savefig("../final_k_analysis_new.png")
print("Saved k-analysis plot to 'final_k_analysis.png'")

# 4. FINALIZE & VISUALIZE CLUSTERS
print(f"\n--- Step 2: Finalizing Results for k={best_k} ---")

# Add cluster labels to original data
df_clean["Cluster"] = best_labels_final

# Show Mean Values per Cluster
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
plt.savefig("../final_clusters_pso_new.png")
print("Saved cluster plot to 'final_clusters_pso.png'")

# Save Data
output_file = "../clustered_uap_metrics_pso_new.csv"
df_clean.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

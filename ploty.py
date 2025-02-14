# Dimensionality reduction with PCA
import numpy as np

pca = PCA(n_components=3)
pca_results = pca.fit_transform(scaled_data)
df_clustering[["PC1", "PC2", "PC3"]] = pca_results
"""
# Determine optimal clusters
best_score = -1
best_n = 3

for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    
    if score > best_score:
        best_score = score
        best_n = n_clusters

print(f"Optimal number of clusters: {best_n} (Silhouette Score: {best_score:.2f})")
"""
best_n = 3
# Final clustering
kmeans = KMeans(n_clusters=best_n, random_state=42)
df_clustering["cluster"] = kmeans.fit_predict(scaled_data)

# Enhanced 3D Visualization
fig = px.scatter_3d(
    df_clustering,
    x="PC1",
    y="PC2",
    z="PC3",
    color="cluster",
    color_continuous_scale=px.colors.qualitative.Vivid,
    hover_data={
        "billing_variability": ":.2f",
        "avg_monthly_spend": ":.2f",
        "engagement_score": ":.2f",
        "usage_to_cost_ratio": ":.2f",
        "plan_type": True,
    },
    title=f"3D Customer Segmentation (Optimal Clusters: {best_n})",
    labels={"cluster": "Segment"},
    width=1200,
    height=800,
    opacity=0.7,
    template="plotly_white",
)

# Add cluster centroids
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

fig.add_trace(
    px.scatter_3d(
        x=centroids_pca[:, 0],
        y=centroids_pca[:, 1],
        z=centroids_pca[:, 2],
        color=np.arange(best_n),
        size=[10] * best_n,
        color_continuous_scale=px.colors.qualitative.Vivid,
    ).data[0]
)

# Add annotations and axis labels
fig.update_layout(
    scene=dict(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        zaxis_title="Principal Component 3",
        xaxis=dict(title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14)),
        zaxis=dict(title_font=dict(size=14)),
    ),
    coloraxis_colorbar=dict(
        title="Cluster",
        tickvals=np.arange(best_n),
        ticktext=[f"Segment {i}" for i in range(best_n)],
    ),
)

# Add rotation animation
fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.6)))

fig.show()

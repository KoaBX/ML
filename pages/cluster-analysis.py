import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="KMeans Cluster Center Analysis", layout="wide")
st.title("KMeans Cluster Centers Analysis")

st.header("Data Input")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()  # Prevents the rest of the app from running until a file is uploaded
    
# 6. Cluster Analysis and Visualization
# Interpret KMeans Clusters
st.subheader("KMeans Cluster Centers Analysis")
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_scaled)
kmeans_centers = kmeans.cluster_centers_
kmeans_cluster_analysis = pd.DataFrame(kmeans_centers, columns=features)
st.dataframe(kmeans_cluster_analysis)

# Visualize feature distribution in each cluster
st.subheader("Cluster Visualization (KMeans)")
# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA results and cluster labels to a DataFrame
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df['Cluster'] = kmeans.labels_

# Select cluster to view
unique_clusters = sorted(pca_df['Cluster'].unique())
selected_cluster = st.selectbox("Select a cluster to visualize (PCA):", unique_clusters)

# Filter for the selected cluster
filtered_pca_df = pca_df[pca_df['Cluster'] == selected_cluster]

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(filtered_pca_df["PC1"], filtered_pca_df["PC2"], alpha=0.7, label=f"Cluster {selected_cluster}")
ax.set_title(f"PCA View of Cluster {selected_cluster}")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()

st.pyplot(fig)


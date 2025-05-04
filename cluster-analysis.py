import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="KMeans Cluster Center Analysis", layout="wide")
st.title("KMeans Cluster Centers Analysis")

# Load or reuse your preprocessed data
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include='number').dropna()
    features = numeric_df.columns
    X_scaled = StandardScaler().fit_transform(numeric_df)

    # Fit KMeans
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X_scaled)
    kmeans_centers = kmeans.cluster_centers_
    kmeans_cluster_analysis = pd.DataFrame(kmeans_centers, columns=features)
    st.dataframe(kmeans_cluster_analysis)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = kmeans.labels_

    selected_cluster = st.selectbox("Select a cluster to visualize (PCA):", sorted(pca_df["Cluster"].unique()))
    filtered_pca = pca_df[pca_df["Cluster"] == selected_cluster]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(filtered_pca["PC1"], filtered_pca["PC2"], alpha=0.7, label=f"Cluster {selected_cluster}")
    ax.set_title(f"PCA View of Cluster {selected_cluster}")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("Please upload a dataset to begin.")

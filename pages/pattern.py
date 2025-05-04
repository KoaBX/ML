
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM, SVC
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import gdown
import itertools

st.set_page_config(page_title="Wind Pattern Analysis", layout="wide")
st.title("Wind Pattern Analysis Dashboard")

# 1. Data Input
st.header("Data Input")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()  # Prevents the rest of the app from running until a file is uploaded

# 4. Modeling
features = [
    'air_pressure', 'air_temp', 'avg_wind_speed', 'max_wind_speed',
    'min_wind_speed', 'relative_humidity',
    'avg_wind_direction_x', 'avg_wind_direction_y',
    'min_wind_direction_x', 'min_wind_direction_y',
    'max_wind_direction_x', 'max_wind_direction_y'
]
df_sample = df[features].sample(n=10000, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Clustering section
st.subheader("Clustering")
clustering_method = st.selectbox("Choose clustering method", [
    "KMeans", "DBSCAN", "GMM", "SVM", "Spectral", "Agglomerative"])

if clustering_method == "KMeans":
    model = KMeans(n_clusters=2, random_state=42).fit(X_scaled)
    labels = model.labels_
    # Cluster evaluation
    st.subheader("Clustering Evaluation")
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
    
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")
        st.write(f"Calinski-Harabasz Score: {ch_score:.2f}")
    else:
        st.warning("Only one cluster detected. Evaluation metrics not available.")
    
    # Visualize clusters using PCA
    fig, axs = plt.subplots(figsize=(8, 5), squeeze=False)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    plt.tight_layout()
    st.pyplot(fig)

elif clustering_method == "DBSCAN":
    model = DBSCAN(eps=1.1, min_samples=10).fit(X_scaled)
    labels = model.labels_
    st.subheader("Clustering Evaluation")
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
    
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")
        st.write(f"Calinski-Harabasz Score: {ch_score:.2f}")
    else:
        st.warning("Only one cluster detected. Evaluation metrics not available.")
    
    # Visualize clusters using PCA
    fig, axs = plt.subplots(figsize=(8, 5), squeeze=False)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    plt.tight_layout()
    st.pyplot(fig)
    
elif clustering_method == "GMM":
    model = GaussianMixture(n_components=2, covariance_type='spherical', random_state=42).fit(X_scaled)
    labels = model.predict(X_scaled)
    st.subheader("Clustering Evaluation")
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
    
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")
        st.write(f"Calinski-Harabasz Score: {ch_score:.2f}")
    else:
        st.warning("Only one cluster detected. Evaluation metrics not available.")
    
    # Visualize clusters using PCA
    fig, axs = plt.subplots(figsize=(8, 5), squeeze=False)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    plt.tight_layout()
    st.pyplot(fig)
    
elif clustering_method == "SVM":
    model = OneClassSVM(kernel='rbf', nu=0.01).fit(X_scaled)
    labels = model.predict(X_scaled)
    labels = [0 if l == -1 else 1 for l in labels]
    st.subheader("Clustering Evaluation")
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
    
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")
        st.write(f"Calinski-Harabasz Score: {ch_score:.2f}")
    else:
        st.warning("Only one cluster detected. Evaluation metrics not available.")

    # Visualize clusters using PCA
    fig, axs = plt.subplots(figsize=(8, 5), squeeze=False)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    plt.tight_layout()
    st.pyplot(fig)

elif clustering_method == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=2)
    labels = model.fit_predict(X_scaled)

    st.subheader("Clustering Evaluation")
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)

        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")
        st.write(f"Calinski-Harabasz Score: {ch_score:.2f}")
    else:
        st.warning("Only one cluster detected. Evaluation metrics not available.")

    # Visualize clusters using PCA
    fig, axs = plt.subplots(figsize=(8, 5), squeeze=False)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    plt.tight_layout()
    st.pyplot(fig)
    
elif clustering_method == "Spectral":
    model = SpectralClustering(n_clusters=2, random_state=42, affinity='nearest_neighbors').fit(X_scaled)
    labels = model.labels_
    st.subheader("Clustering Evaluation")
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
    
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Score: {db_score:.2f}")
        st.write(f"Calinski-Harabasz Score: {ch_score:.2f}")
    else:
        st.warning("Only one cluster detected. Evaluation metrics not available.")
    
    # Visualize clusters using PCA
    fig, axs = plt.subplots(figsize=(8, 5), squeeze=False)
    axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    plt.tight_layout()
    st.pyplot(fig)

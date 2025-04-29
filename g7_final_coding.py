import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns

st.title("🌬️ Wind Pattern Clustering Dashboard")

uploaded_file = st.file_uploader("Upload your wind dataset (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    features = ['air_pressure', 'air_temp', 'avg_wind_speed', 'max_wind_speed',
                'min_wind_speed', 'rain_duration', 'relative_humidity',
                'avg_wind_direction', 'min_wind_direction', 'max_wind_direction']
    df = df.dropna(subset=features)

    for col in ['avg_wind_direction', 'min_wind_direction', 'max_wind_direction']:
        df[f"{col}_x"] = np.cos(np.radians(df[col]))
        df[f"{col}_y"] = np.sin(np.radians(df[col]))
        df.drop(columns=[col], inplace=True)

    model_features = [col for col in df.columns if col.endswith('_x') or col.endswith('_y') or col in features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[model_features])

    st.subheader("📉 PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    st.subheader("🔍 Clustering Model Selection")
    model_choice = st.selectbox("Choose Clustering Model", ["KMeans", "DBSCAN", "Hierarchical"])

    if model_choice == "KMeans":
        k = st.slider("Number of clusters (k)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
    elif model_choice == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
        min_samples = st.slider("Min Samples", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        k = st.slider("Number of clusters", 2, 10, 3)
        model = AgglomerativeClustering(n_clusters=k)

    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    st.markdown(f"**Detected Clusters:** {n_clusters}")
    
    if n_clusters >= 2:
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
    
        st.metric("Silhouette Score", f"{sil_score:.4f}")
        st.metric("Davies-Bouldin Index", f"{db_score:.4f}")
        st.metric("Calinski-Harabasz Index", f"{ch_score:.4f}")
    else:
        st.warning("⚠️ Less than 2 clusters detected — clustering metrics not applicable.")

    st.subheader("🧭 Cluster Visualization (PCA)")
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10')
    st.pyplot(fig)

    st.subheader("🔥 Feature Relationships")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[model_features].corr(), cmap="coolwarm", annot=True, ax=ax_corr)
    st.pyplot(fig_corr)
else:
    st.info("👆 Please upload a CSV file to get started.")

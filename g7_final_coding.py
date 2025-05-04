# Update the Streamlit code to include sections for:
# 1. Data input
# 2. Data preprocessing
# 3. EDA, analysis, and visualization
# 4. Modeling (with tuned parameters from notebook if available)
# 5. Result visualization
# 6. Cluster analysis with dropdown

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

st.dataframe(df.head())

# Clean column names by stripping spaces
df.columns = df.columns.str.strip()

#Timestamp processing
df['hpwren_timestamp'] = pd.to_datetime(df['hpwren_timestamp'])

# Extract useful fe from the timestamp
df['year'] = df['hpwren_timestamp'].dt.year
df['month'] = df['hpwren_timestamp'].dt.month
df['day'] = df['hpwren_timestamp'].dt.day
df['hour'] = df['hpwren_timestamp'].dt.hour
df['minute'] = df['hpwren_timestamp'].dt.minute
df['second'] = df['hpwren_timestamp'].dt.second
df['timestamp'] = pd.to_datetime(df['hpwren_timestamp']).dt.time
df.rename(columns={'hpwren_timestamp': 'full_timestamp'}, inplace=True)

# Check for missing values
missing_values = df.isnull().sum()
df = df.dropna()

# Check for duplicate rows
duplicated_rows = df.duplicated().sum()

df['rain_accumulation'] = df['rain_duration'].apply(lambda x: 1 if x != 0 else 0)

outlier_cols = [
    'air_pressure', 'air_temp',
    'avg_wind_direction', 'avg_wind_speed',
    'max_wind_direction', 'max_wind_speed',
    'min_wind_direction', 'min_wind_speed',
    'relative_humidity'
]
plt.figure(figsize=(12, 6))
df[[col for col in outlier_cols if col != 'rain_duration']].boxplot(rot=45)
plt.title("Boxplot of Potential Outlier Columns")
plt.show()

non0_rain = df.loc[df['rain_duration'] != 0, 'rain_duration']
plt.figure(figsize=(12, 6))
non0_rain.plot.box(rot=45)
plt.title("Boxplot of Rain Duration")
plt.show()

# 1. Outlier detection for general columns using IQR
Q1 = df[outlier_cols].quantile(0.25)
Q3 = df[outlier_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers for each column in outlier_cols
outliers_count = ((df[outlier_cols] < lower_bound) | (df[outlier_cols] > upper_bound)).sum()
print("Outliers count for general columns:\n", outliers_count)

# Outlier detection for rain_duration using z-score (for nonzero values only)
non0_rain = df.loc[df['rain_duration'] != 0, 'rain_duration']
z_scores = stats.zscore(non0_rain)
threshold = 3
non0_outliers_mask = np.abs(z_scores) > threshold
rain_duration_z_outliers_count = non0_outliers_mask.sum()
print("Outlier count for rain_duration (nonzero values, using zscore):", rain_duration_z_outliers_count)

# Get the indices of nonzero rain_duration outliers
non0_outlier_indices = non0_rain.index[non0_outliers_mask]

print("Initial DataFrame shape:", df.shape)
# Remove rows with outliers in general columns (using IQR)
df = df[~((df[outlier_cols] < lower_bound) | (df[outlier_cols] > upper_bound)).any(axis=1)]

# Remove rows where rain_duration is nonzero and flagged as an outlier based on z-score
df = df[~((df['rain_duration'] != 0) & (df.index.isin(non0_outlier_indices)))]

print("DataFrame shape after outlier removal:", df.shape)

# Convert wind direction columns into x, y components
for col in ['avg_wind_direction', 'min_wind_direction', 'max_wind_direction']:
    df[f'{col}_x'] = np.cos(np.radians(df[col]))
    df[f'{col}_y'] = np.sin(np.radians(df[col]))


# Print the transformed dataset
print(df.head())


# Recover wind direction angles from x, y
df['avg_wind_direction_deg'] = np.degrees(np.arctan2(df['avg_wind_direction_y'], df['avg_wind_direction_x'])) % 360
df['min_wind_direction_deg'] = np.degrees(np.arctan2(df['min_wind_direction_y'], df['min_wind_direction_x'])) % 360
df['max_wind_direction_deg'] = np.degrees(np.arctan2(df['max_wind_direction_y'], df['max_wind_direction_x'])) % 360


print(df[['avg_wind_direction_deg', 'min_wind_direction_deg', 'max_wind_direction_deg']].head())

# Columns to scale (excluding wind direction)
features_to_scale = ['air_pressure', 'air_temp', 'avg_wind_speed',
                     'max_wind_speed', 'min_wind_speed',
                     'rain_duration', 'relative_humidity']

# Apply MinMaxScaler
scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
df.describe()

# Function to categorize wind direction into N, NE, E, SE, S, SW, W, NW
def categorize_wind_direction(degrees):
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]  # Corrected bin edges
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return pd.cut(degrees % 360, bins=bins, labels=labels, include_lowest=True)

# Apply categorization
df['avg_wind_category'] = categorize_wind_direction(df['avg_wind_direction'])
df['max_wind_category'] = categorize_wind_direction(df['max_wind_direction'])
df['min_wind_category'] = categorize_wind_direction(df['min_wind_direction'])

# Count occurrences
avg_counts = df['avg_wind_category'].value_counts().reindex(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fill_value=0)
max_counts = df['max_wind_category'].value_counts().reindex(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fill_value=0)
min_counts = df['min_wind_category'].value_counts().reindex(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fill_value=0)

# Create figure with 2 rows (3 bar charts on top, 3 polar plots below)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Bar charts (Top Row)
axes[0, 0].bar(avg_counts.index, avg_counts.values, color='blue')
axes[0, 0].set_title("Average Wind Direction")
axes[0, 0].set_xlabel("Direction")
axes[0, 0].set_ylabel("Count")

axes[0, 1].bar(max_counts.index, max_counts.values, color='red')
axes[0, 1].set_title("Max Wind Direction")
axes[0, 1].set_xlabel("Direction")

axes[0, 2].bar(min_counts.index, min_counts.values, color='green')
axes[0, 2].set_title("Min Wind Direction")
axes[0, 2].set_xlabel("Direction")

# Polar plots (Bottom Row)
angles_avg = np.deg2rad(df['avg_wind_direction'])  # Convert to radians
angles_max = np.deg2rad(df['max_wind_direction'])
angles_min = np.deg2rad(df['min_wind_direction'])

# Average Wind Rose
ax1 = fig.add_subplot(2, 3, 4, projection='polar')
ax1.hist(angles_avg, bins=36, color='blue', alpha=0.75)
ax1.set_title("Wind Rose - Average Wind Direction")

# Max Wind Rose
ax2 = fig.add_subplot(2, 3, 5, projection='polar')
ax2.hist(angles_max, bins=36, color='red', alpha=0.75)
ax2.set_title("Wind Rose - Max Wind Direction")

# Min Wind Rose
ax3 = fig.add_subplot(2, 3, 6, projection='polar')
ax3.hist(angles_min, bins=36, color='green', alpha=0.75)
ax3.set_title("Wind Rose - Min Wind Direction")

# Adjust layout
plt.tight_layout()
plt.show()

plt.hist(df['avg_wind_direction_deg'], bins=36, edgecolor='black', alpha=0.7)  # 10° bins
plt.xlabel("Wind Direction (Degrees)")
plt.ylabel("Frequency")
plt.title("Wind Direction Distribution")
plt.show()

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

# Convert X_scaled to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Export to CSV
X_scaled_df.to_csv("X_scaled_output.csv", index=False)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def evaluate_model(X, labels, name):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return (name, n_clusters, -1.0, -1.0, -1.0)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return (name, n_clusters, sil, db, ch)
    

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
    "KMeans", "DBSCAN", "Agglomerative", "GMM", "SVM", "Spectral"])

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
    
elif clustering_method == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=2).fit(X_scaled)
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

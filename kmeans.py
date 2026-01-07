import numpy as np
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from google.colab import files
import zipfile

# 1) FILE PICKER 
print("Please upload your CSV")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(" File uploaded:", filename)
# Handle ZIP files
if filename.endswith('.zip'):
    with zipfile.ZipFile(io.BytesIO(uploaded[filename]), 'r') as zf:
        # Assuming the CSV is the first file in the zip or named similarly
        csv_filename = [name for name in zf.namelist() if name.endswith('.csv')][0]
        with zf.open(csv_filename) as csv_file:
            df = pd.read_csv(csv_file, encoding='latin1')
    print(f"CSV '{csv_filename}' extracted from ZIP and loaded.")
else:
    # load the CSV directly
    df = pd.read_csv(io.BytesIO(uploaded[filename]), encoding='latin1')
print("Dataset Loaded. Shape:", df.shape)

# 2) SELECT IMPORTANT COLUMNS
columns_needed = ['product_category_tree',
                  'retail_price',
                  'discounted_price',
                  'brand',
                  'description']
df_use = df[columns_needed].copy()
df_use.dropna(inplace=True)
print("After Cleaning:", df_use.shape)

# 3) FEATURE ENGINEERING
# NUMERIC FEATURES 
num_features = df_use[['retail_price', 'discounted_price']].astype(float)
#  BRAND ONE-HOT ENCODING 
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
brand_encoded = ohe.fit_transform(df_use[['brand']])
# TEXT FEATURES (DESCRIPTION) 
tfidf = TfidfVectorizer(max_features=300)
text_vec = tfidf.fit_transform(df_use['description'].astype(str)).toarray()

# 4) COMBINE FEATURES
X = np.hstack([num_features, brand_encoded, text_vec])
print("Final Feature Vector Shape:", X.shape)

# 5) SCALING + PCA
scaler = StandardScaler()
scaled = scaler.fit_transform(X)
pca = PCA(n_components=15)
reduced = pca.fit_transform(scaled)
print("PCA Reduced Shape:", reduced.shape)

# 6) CHAMELEON-LIKE CLUSTERING (HDBSCAN)
cham = hdbscan.HDBSCAN(min_cluster_size=40, metric='euclidean')
labels_cham = cham.fit_predict(reduced)
print("Chameleon Cluster Labels:", np.unique(labels_cham))
# Safe silhouette (only if clusters exist)
unique_cham = np.unique(labels_cham)
if len(unique_cham) > 1 and not (len(unique_cham) == 1 and unique_cham[0] == -1):
    mask = labels_cham != -1
    sil_cham = silhouette_score(reduced[mask], labels_cham[mask])
else:
    sil_cham = 0.0

# 7) K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=5, random_state=42)
labels_km = kmeans.fit_predict(reduced)
sil_km = silhouette_score(reduced, labels_km)

# 8) SHOW SILHOUETTE SCORES
print("\nSilhouette Scores:")
print("K-Means:", sil_km)
print("Chameleon (HDBSCAN):", sil_cham)

# 9) HEATMAP (K-MEANS)
df_heat = pd.DataFrame(reduced)
df_heat["cluster"] = labels_km
sns.clustermap(df_heat.iloc[:, :-1], cmap="coolwarm")
plt.title("Product Cluster Heatmap (K-Means)")
plt.show()

# 10) SCATTER PLOTS
plt.figure(figsize=(12, 5))
# K-MEANS
plt.subplot(1, 2, 1)
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_km, s=10)
plt.title("K-Means Clustering (PCA 2D)")
# CHAMELEON
plt.subplot(1, 2, 2)
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_cham, s=10)
plt.title("Chameleon-like Clustering (HDBSCAN)")
plt.tight_layout()
plt.show()

# 11) FINAL SUMMARY
print("\n-------- FINAL SUMMARY --------")
print(f"K-Means Silhouette: {sil_km:.4f}")
print(f"Chameleon Silhouette: {sil_cham:.4f}")
if sil_km > sil_cham:
    print("Conclusion: K-Means produced more well-defined clusters.")
else:
    print("Conclusion: Chameleon-like clustering performed better.")

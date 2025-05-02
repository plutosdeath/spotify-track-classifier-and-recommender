import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Load dataset
df = pd.read_csv("SpotifyAudioFeaturesApril2019.csv")
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters with genre-like tags
cluster_labels = {
    0: "Pop",
    1: "Dance / Hip-Hop",
    2: "Ambient / Classical",
    3: "Rock / Electronic",
    4: "Indie / Alt"
}
df['genre_label'] = df['cluster'].map(cluster_labels)

# Save everything for the app
df.to_csv("clustered_spotify_data.csv", index=False)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("✅ Model, scaler, and dataset saved.")
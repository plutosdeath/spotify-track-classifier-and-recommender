# Spotify Audio Genre Classifier & Recommender

## Features

- 🔍 Clusters over 23,000 Spotify tracks using K-Means based on audio features
- 🎼 Assigns intuitive genre labels to clusters (e.g., "Pop", "Ambient / Classical")
- 🎛️ Streamlit app lets users enter custom track features
- 🎵 Predicts genre and recommends similar songs
- 📈 Includes visual analysis: heatmaps, PCA, radar charts (optional)

---

## Technologies Used

- **Python** (Pandas, Scikit-learn, Matplotlib, Seaborn)
- **Machine Learning**: K-Means Clustering, StandardScaler
- **Streamlit** for the interactive UI
- **Spotify Audio Features Dataset** (April 2019)

## Clustering Logic

The model uses 11 core audio features from Spotify:

- `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

These features are scaled and clustered into **5 groups**, each mapped to a high-level genre category based on average values and musical patterns.

| Cluster | Label              | Description                                                |
|---------|-------------------|------------------------------------------------------------|
| 0       | Pop               | Balanced tracks with moderate energy and valence          |
| 1       | Dance / Hip-Hop   | High energy, high danceability, speech-heavy              |
| 2       | Ambient / Classical | Quiet, instrumental, acoustic, low tempo                |
| 3       | Rock / Electronic | High energy and major key, strong rhythmic characteristics |
| 4       | Indie / Alt       | Minor key, medium energy, introspective tones             |

---

## Try It Out

Run the Streamlit app locally:

```bash
streamlit run streamlit_app.py
```

## Screenshots

![image](https://github.com/user-attachments/assets/3faef693-710e-4b56-ac1d-ef0274701c3b)

![image](https://github.com/user-attachments/assets/9363f1fa-10d6-49a0-88e4-6f7187df7ed5)


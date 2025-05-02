import streamlit as st
import pandas as pd
import pickle

# Load preprocessed data, model, and scaler
df = pd.read_csv("clustered_spotify_data.csv")
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("kmeans_model.pkl", "rb"))

st.set_page_config(page_title="🎵 Spotify Genre Classifier", layout="centered")
st.title("🎧 Spotify Audio Genre Classifier")
st.subheader("Enter audio features to classify a track and get recommendations:")

# User input sliders
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
key = st.slider("Key", 0, 11, 5)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
mode = st.radio("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.2)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1)
liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 50, 200, 120)

# Create dataframe for prediction
input_data = pd.DataFrame([{
    'danceability': danceability,
    'energy': energy,
    'key': key,
    'loudness': loudness,
    'mode': mode,
    'speechiness': speechiness,
    'acousticness': acousticness,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'valence': valence,
    'tempo': tempo
}])

# Predict cluster and genre
input_scaled = scaler.transform(input_data)
cluster = model.predict(input_scaled)[0]

genre_map = {
    0: "Pop",
    1: "Dance / Hip-Hop",
    2: "Ambient / Classical",
    3: "Rock / Electronic",
    4: "Indie / Alt"
}
predicted_genre = genre_map[cluster]

st.markdown(f"### 🎶 Predicted Genre: **{predicted_genre}**")

# Recommend 5 similar tracks from same genre
st.markdown("### 🔁 Recommended Tracks:")
recommendations = df[df['cluster'] == cluster].sample(5)[['track_name', 'artist_name']]

st.table(recommendations.rename(columns={
    'track_name': 'Track Name',
    'artist_name': 'Artist'
}))
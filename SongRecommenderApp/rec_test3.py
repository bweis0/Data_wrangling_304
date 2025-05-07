import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import numpy as np
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# --- Function to load file regardless of PyInstaller or dev mode ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # For PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Load dataset ---
csv_path = resource_path("SpotifyFeatures.csv")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["track_name", "artist_name", "genre"])
df = df.drop_duplicates(subset=["track_name", "artist_name"])

# Normalize features
audio_features = ["danceability", "energy", "tempo", "valence", "acousticness", "loudness"]
scaler = StandardScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# --- Find matching song ---
def find_song(title, artist=None):
    title = title.lower()
    artist = artist.lower() if artist else None
    if artist:
        match = df[
            df["track_name"].str.lower().str.contains(title) &
            df["artist_name"].str.lower().str.contains(artist)
        ]
    else:
        match = df[df["track_name"].str.lower().str.contains(title)]
    return match.iloc[0] if not match.empty else None

# --- Recommendation logic ---
def get_recommendations(title, artist, method, genre_filter):
    song = find_song(title, artist)
    if song is None:
        return None, []
    filtered_df = df[df["genre"] == song["genre"]].copy() if genre_filter else df.copy()
    song_vec = song[audio_features].values.reshape(1, -1)

    if method == "Cosine":
        sim = cosine_similarity(filtered_df[audio_features], song_vec).flatten()
        filtered_df["similarity"] = sim
    elif method == "Euclidean":
        dist = euclidean_distances(filtered_df[audio_features], song_vec).flatten()
        sim = 1 / (1 + dist)
        filtered_df["similarity"] = sim
    else:
        return song, []

    filtered_df = filtered_df[filtered_df["track_name"] != song["track_name"]]
    recs = filtered_df.sort_values(by="similarity", ascending=False).head(5)
    return song, recs[["track_name", "artist_name", "genre", "similarity"]]

# --- GUI Setup ---
root = tk.Tk()
root.title("🎧 Song Recommender")
root.geometry("600x520")

input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="Song Title:").grid(row=0, column=0, sticky="e")
title_entry = tk.Entry(input_frame, width=40)
title_entry.grid(row=0, column=1, padx=5)

tk.Label(input_frame, text="Artist (optional):").grid(row=1, column=0, sticky="e")
artist_entry = tk.Entry(input_frame, width=40)
artist_entry.grid(row=1, column=1, padx=5)

tk.Label(input_frame, text="Similarity Method:").grid(row=2, column=0, sticky="e")
method_var = tk.StringVar(value="Cosine")
method_dropdown = ttk.Combobox(input_frame, textvariable=method_var, values=["Cosine", "Euclidean"])
method_dropdown.grid(row=2, column=1, padx=5, pady=5)

genre_var = tk.BooleanVar()
tk.Checkbutton(input_frame, text="Match Genre Only", variable=genre_var).grid(row=3, column=1, sticky="w")

open_input_var = tk.BooleanVar()
tk.Checkbutton(input_frame, text="Open song input in Spotify", variable=open_input_var).grid(row=4, column=1, sticky="w")

# --- Results and status ---
results_box = tk.Listbox(root, width=80, height=10)
results_box.pack(pady=10)

status_label = tk.Label(root, text="", font=("Arial", 10), fg="blue")
status_label.pack()

recommendations = []

# --- Handle song click ---
def on_select(event):
    if not recommendations:
        return
    index = results_box.curselection()
    if index:
        i = index[0] - 1  # Skip the label row
        if 0 <= i < len(recommendations):
            track, artist = recommendations[i][:2]
            query = f"{track} {artist}".replace(" ", "%20")
            url = f"https://open.spotify.com/search/{query}"
            webbrowser.open(url)

results_box.bind("<<ListboxSelect>>", on_select)

# --- Recommend button ---
def run_recommender():
    results_box.delete(0, tk.END)
    song_title = title_entry.get().strip()
    artist_name = artist_entry.get().strip()
    method = method_var.get()
    use_genre = genre_var.get()

    song, recs = get_recommendations(song_title, artist_name, method, use_genre)

    if song is None:
        status_label.config(text="")
        messagebox.showerror("Not Found", "Song not found in dataset.")
        return

    # Show user what was matched
    status_label.config(
        text=f"Using: {song['track_name']} by {song['artist_name']} (Genre: {song['genre']})"
    )

    # Open song if box is checked
    if open_input_var.get():
        q = f"{song['track_name']} {song['artist_name']}".replace(" ", "%20")
        url = f"https://open.spotify.com/search/{q}"
        webbrowser.open(url)

    global recommendations
    recommendations = recs.values.tolist()

    results_box.insert(tk.END, "Top 5 Recommendations:")
    for i, (track, artist, genre, sim) in enumerate(recommendations, 1):
        results_box.insert(tk.END, f"{i}. {track} by {artist} (Genre: {genre}) - Score: {round(sim, 3)}")
    results_box.insert(tk.END, "Click a song to open it in Spotify.")

tk.Button(root, text="Recommend Songs", command=run_recommender, bg="#4CAF50", fg="white").pack(pady=10)

root.mainloop()

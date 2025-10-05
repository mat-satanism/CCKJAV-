import json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RAW_FEATURES = ["koi_period","koi_duration","koi_depth","koi_prad",
                "koi_steff","koi_slogg","koi_srad","koi_kepmag"]

def load_data(path):
    df = pd.read_csv(path)
    X = df[RAW_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def train_autoencoder(Xs, latent_dim=4, epochs=50):
    n_in = Xs.shape[1]
    inp = keras.layers.Input(shape=(n_in,))
    enc = keras.layers.Dense(16, activation="relu")(inp)
    enc = keras.layers.Dense(latent_dim, activation="relu")(enc)
    dec = keras.layers.Dense(16, activation="relu")(enc)
    out = keras.layers.Dense(n_in, activation=None)(dec)
    ae = keras.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(Xs, Xs, epochs=epochs, batch_size=32, verbose=1)
    enc_model = keras.Model(inp, enc)
    return ae, enc_model

def main():
    Xs, scaler = load_data("data/cleaned/KOI_clean_for_learning.csv")
    ae, encoder = train_autoencoder(Xs)
    latent = encoder.predict(Xs)

    km = KMeans(n_clusters=3, random_state=42)
    clusters = km.fit_predict(latent)

    Path("artifacts_ae").mkdir(exist_ok=True)
    ae.save("artifacts_ae/autoencoder.h5")
    encoder.save("artifacts_ae/encoder.h5")
    np.save("artifacts_ae/kmeans_centers.npy", km.cluster_centers_)
    json.dump({"n_clusters": 3}, open("artifacts_ae/config.json", "w"))

    print("Training complete, clusters saved → artifacts_ae/")

if __name__ == "__main__":
    main()

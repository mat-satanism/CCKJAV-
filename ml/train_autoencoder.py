import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import os

def build_autoencoder(input_dim, encoding_dim=8):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    return autoencoder, encoder


def main():
    base_dir = Path(__file__).resolve().parents[1]

    data_path = base_dir / "data" / "cleaned" / "KOI_clean_for_learning.csv"
    out_dir = base_dir / "artifacts_ae"
    preds_dir = base_dir / "preds"

    out_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    features = [
        "koi_period", "koi_duration", "koi_depth", "koi_prad",
        "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
    ]
    print(f"Data loaded: {df.shape[0]} rows, {len(features)} features")

    X = df[features].fillna(df[features].mean())
    X_scaled = StandardScaler().fit_transform(X)

    input_dim = X_scaled.shape[1]
    ae, encoder = build_autoencoder(input_dim)
    ae.compile(optimizer=Adam(1e-3), loss='mse')

    print("Training Autoencoder...")
    ae.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=1)

    encoder.save(out_dir / "encoder_model.h5")
    ae.save(out_dir / "autoencoder_full.h5")

    print("Performing KMeans clustering...")
    X_encoded = encoder.predict(X_scaled)
    km = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = km.fit_predict(X_encoded)
    df["cluster_dist"] = np.min(km.transform(X_encoded), axis=1)

    df.to_csv(preds_dir / "toi_ae_clusters.csv", index=False)
    print(f"Clusters saved to {preds_dir / 'toi_ae_clusters.csv'}")

    report = {"cluster_centers": km.cluster_centers_.tolist()}
    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out_dir / 'eval_report.json'}")

    print("\nAutoencoder training and clustering completed successfully!")


if __name__ == "__main__":
    main()

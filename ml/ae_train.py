from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import dump
from tensorflow import keras
from tensorflow.keras import layers

RAW_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]


def build_autoencoder(input_dim: int, encoding_dim: int = 8) -> tuple[keras.Model, keras.Model]:
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inp)
    code = layers.Dense(encoding_dim, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(code)
    # Linear output because inputs are standardized (zero mean / unit var)
    out = layers.Dense(input_dim, activation=None)(x)
    ae = keras.Model(inp, out)
    enc = keras.Model(inp, code)
    return ae, enc


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "cleaned" / "KOI_clean_for_learning.csv"
    out_dir = base_dir / "artifacts_ae"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df[RAW_FEATURES].copy()
    X_filled = X.astype(float).replace([np.inf, -np.inf], np.nan)
    X_filled = X_filled.fillna(X_filled.mean())

    scaler = StandardScaler().fit(X_filled.values)
    Xn = scaler.transform(X_filled.values)

    ae, enc = build_autoencoder(Xn.shape[1], encoding_dim=8)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    ae.fit(Xn, Xn, epochs=50, batch_size=32, shuffle=True, verbose=1)

    Z = enc.predict(Xn, verbose=0)

    N_CLUST = 5
    km = KMeans(n_clusters=N_CLUST, n_init="auto", random_state=42)
    km.fit(Z)

    ae.save(out_dir / "autoencoder_full.h5")
    enc.save(out_dir / "encoder_model.h5")
    dump(scaler, out_dir / "scaler.joblib")
    dump(km, out_dir / "kmeans.joblib")

    (out_dir / "config.json").write_text(json.dumps({
        "n_clusters": N_CLUST,
        "features": RAW_FEATURES,
        "encoding_dim": 8
    }, indent=2), encoding="utf-8")

    labels, counts = np.unique(km.labels_, return_counts=True)
    eval_report = {
        "cluster_sizes": {int(l): int(c) for l, c in zip(labels, counts)},
        "cluster_label_map": {}
    }
    (out_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    print("[AE] Artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()
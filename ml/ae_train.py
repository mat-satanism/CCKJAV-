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

RAW8 = ["koi_period","koi_duration","koi_depth","koi_prad",
        "koi_steff","koi_slogg","koi_srad","koi_kepmag"]

def build_autoencoder(n_in: int, code_dim: int = 8):
    inp = layers.Input(shape=(n_in,))
    x = layers.Dense(16, activation="relu")(inp)
    code = layers.Dense(code_dim, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(code)
    out = layers.Dense(n_in, activation=None)(x)  # лінійний вихід (бо standardized)
    return keras.Model(inp, out), keras.Model(inp, code)

def main():
    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(base / "data/cleaned/KOI_clean_for_learning.csv")
    X = df[RAW8].astype(float).replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.mean())

    scaler = StandardScaler().fit(X.values)
    Xn = scaler.transform(X.values)

    ae, enc = build_autoencoder(Xn.shape[1], code_dim=8)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    ae.fit(Xn, Xn, epochs=50, batch_size=32, shuffle=True, verbose=1)

    Z = enc.predict(Xn, verbose=0)
    km = KMeans(n_clusters=5, n_init="auto", random_state=42).fit(Z)

    out = base / "artifacts_ae"; out.mkdir(exist_ok=True)
    ae.save(out / "autoencoder_full.h5")
    enc.save(out / "encoder_model.h5")
    dump(scaler, out / "scaler.joblib")
    dump(km, out / "kmeans.joblib")
    (out / "config.json").write_text(json.dumps({"n_clusters":5,"features":RAW8}, indent=2))

    print("[AE] saved →", out)

if __name__ == "__main__":
    main()

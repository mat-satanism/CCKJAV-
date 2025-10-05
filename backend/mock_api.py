from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import io, numpy as np, uvicorn

CLASSES = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class OneInput(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_steff: float
    koi_slogg: float
    koi_srad: float
    koi_kepmag: float

def softmax(x):
    e = np.exp(x - np.max(x))
    return (e / e.sum()).tolist()

def mock_scores(row: dict) -> list[float]:
    score_planet = 0.002*row["koi_depth"] + 0.5*row["koi_duration"] + 0.3*row["koi_prad"]
    score_fp = 0.2*(abs(row["koi_slogg"]-3.0)) + 0.001*(row["koi_kepmag"]-12)**2
    score_cand = 0.5
    return softmax(np.array([score_planet, score_cand, score_fp]))

@app.post("/predict_one")
def predict_one(inp: OneInput):
    p = mock_scores(inp.dict())
    cls = CLASSES[int(np.argmax(p))]
    return {"pred_class": cls,
            "probs": {CLASSES[i]: round(float(p[i]), 4) for i in range(3)}}

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))  # читаємо CSV

    probs, labels = [], []
    for _, r in df.iterrows():
        p = mock_scores(r.to_dict())
        probs.append(p)
        labels.append(CLASSES[int(np.argmax(p))])

    out = df.copy()
    out["pred_class"] = labels
    out["prob_CONFIRMED"] = [p[0] for p in probs]
    out["prob_CANDIDATE"] = [p[1] for p in probs]
    out["prob_FALSE POSITIVE"] = [p[2] for p in probs]

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    return Response(content=csv_bytes, media_type="text/csv")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
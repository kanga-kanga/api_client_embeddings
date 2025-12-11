import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import requests
import uvicorn

app = FastAPI(title="Embedding API", version="1.0.0")

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_PREFIX = os.getenv("EMBED_PREFIX", "").strip()
# Mode remote (Hugging Face Inference API)
USE_REMOTE = os.getenv("EMBED_USE_REMOTE", "0").strip() == "1"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_ENDPOINT = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
try:
    MAX_SEQ_LEN = int(os.getenv("EMBED_MAX_SEQ_LEN", "256"))
except ValueError:
    MAX_SEQ_LEN = 256

# Lazy-load: ne charge le modèle qu'à la première requête d'embed.
model = None
def get_model():
    if USE_REMOTE:
        return None
    global model
    if model is None:
        try:
            model = SentenceTransformer(MODEL_NAME, device="cpu")
            try:
                model.max_seq_length = MAX_SEQ_LEN
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle {MODEL_NAME}: {e}")
    return model


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "loaded": (not USE_REMOTE) and bool(model is not None),
        "remote": USE_REMOTE,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    t = (req.text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="Texte vide")
    try:
        inp = f"{EMBED_PREFIX}{t}" if EMBED_PREFIX else t
        if USE_REMOTE:
            if not HF_TOKEN:
                raise HTTPException(status_code=500, detail="HF_TOKEN manquant pour EMBED_USE_REMOTE=1")
            headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
            # HF Inference API 'feature-extraction' renvoie une matrice [tokens][dim];
            # Beaucoup de ST endpoint renvoient directement l'embedding agrégé.
            # On envoie une liste de textes pour compatibilité.
            try:
                resp = requests.post(HF_ENDPOINT, headers=headers, json={"inputs": [inp]}, timeout=120)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"HF API error: {resp.text}")
                data = resp.json()
                # Normaliser: si l'API renvoie [embedding] ou [[embedding]]
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                    arr = data[0]
                else:
                    arr = data
                # Aplatir si nécessaire
                if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
                    # Certains endpoints renvoient par token: moyenne simple
                    import numpy as np
                    arr = np.array(arr, dtype=float).mean(axis=0).tolist()
                return {"embedding": arr, "dim": len(arr)}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Erreur HF remote: {e}")
        else:
            mdl = get_model()
            with torch.inference_mode():
                vec = mdl.encode([inp], normalize_embeddings=False)
            arr = vec[0].tolist()
            return {"embedding": arr, "dim": len(arr)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur embedding: {e}")


if __name__ == "__main__":
    # Lecture des valeurs de configuration et lancement automatique
    host = os.getenv("EMBED_HOST", "0.0.0.0")
    port_str = os.getenv("EMBED_PORT", "8080")
    try:
        port = int(port_str)
    except ValueError:
        raise RuntimeError(f"EMBED_PORT invalide: '{port_str}' doit être un entier")

    # Démarre le serveur Uvicorn avec les valeurs lues
    print(f"[EmbeddingAPI] Démarrage sur http://{host}:{port} | modèle={MODEL_NAME}")
    # Utiliser l'objet app directement pour éviter les problèmes d'import de module
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

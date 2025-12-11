import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI(title="Embedding API", version="1.0.0")

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle {MODEL_NAME}: {e}")


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    t = (req.text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="Texte vide")
    try:
        vec = model.encode([t], normalize_embeddings=False)
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

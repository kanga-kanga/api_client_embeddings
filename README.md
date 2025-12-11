# Embedding API (FastAPI + SentenceTransformers)

Petite API pour générer des embeddings avec le modèle en ligne
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Endpoints
- GET `/health`: statut + modèle chargé
- POST `/embed`: body `{ "text": "..." }` → `{ embedding: number[], dim: 384 }`

## Variables d'environnement
- `EMBED_MODEL` (optionnel): par défaut `intfloat/multilingual-e5-small` (léger et multilingue)
- `EMBED_USE_REMOTE` (optionnel): `1` pour utiliser l'API d'inférence Hugging Face (pas de téléchargement local), `0` par défaut
- `HF_TOKEN` (optionnel mais requis si `EMBED_USE_REMOTE=1`): token Hugging Face
- `HF_HUB_ENABLE_HF_TRANSFER=1` (optionnel): accélère le téléchargement du modèle
- `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1` (optionnel): évite imports/logs TensorFlow/Flax
- `EMBED_HOST` (optionnel): hôte de binding (défaut `0.0.0.0`)
- `EMBED_PORT` (optionnel): port (défaut `8080`)
 - `EMBED_MAX_SEQ_LEN` (optionnel): longueur max des séquences pour réduire la mémoire (défaut `256`)
 - `EMBED_PREFIX` (optionnel): préfixe pour certaines familles de modèles (ex: `query: ` pour E5)

## Lancer en local (PowerShell)
```pwsh
python -m pip install -r .\requirements.txt
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"   # optionnel
python .\app.py
```

## Tester (PowerShell)
- Santé
```pwsh
Invoke-RestMethod -Uri "http://127.0.0.1:8080/health"
```
- Embedding (UTF-8 recommandé en PowerShell)
```pwsh
$payload = @{ text = "Une vie de défis" }
$body = [System.Text.Encoding]::UTF8.GetBytes(($payload | ConvertTo-Json -Compress))
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/embed" -ContentType "application/json; charset=utf-8" -Body $body
```
- Ou via curl
```pwsh
curl -X POST "http://127.0.0.1:8080/embed" -H "Content-Type: application/json; charset=utf-8" -d '{"text":"Une vie de défis"}'
```

## Déploiement Render (Blueprint) — Repo dédié
Ce dossier est prêt pour être la racine d'un dépôt Git (Option B).

Contenus nécessaires à la racine du dépôt:
- `app.py`
- `requirements.txt`
- `render.yaml`

Étapes:
1) Pousser ce dossier dans un nouveau repo GitHub
2) Render → New → Blueprint → sélectionner le repo → Deploy
3) Variables d'environnement côté service:
  - `EMBED_MODEL=intfloat/multilingual-e5-small`
  - `EMBED_PREFIX="query: "` (recommandé pour E5)
  - `EMBED_USE_REMOTE=0` (mettre `1` pour passer par Hugging Face Inference API)
  - `HF_TOKEN=<ton_token>` (requis si `EMBED_USE_REMOTE=1`)
   - `HF_HUB_ENABLE_HF_TRANSFER=1`
   - `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1` (optionnel)

Le premier démarrage télécharge le modèle. Les suivants réutilisent le cache.

### Health et test après déploiement
```bash
curl https://<ton-service>.onrender.com/health
curl -X POST https://<ton-service>.onrender.com/embed \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"text":"Une vie de défis"}'
```

## Notes
- PyTorch (CPU) est installé via `requirements.txt`.
- Pour Flutter Web, si besoin de CORS permissif, on peut l'ajouter facilement (FastAPI middleware).
- Dimension des embeddings attendue: `384` (MiniLM L12 v2).
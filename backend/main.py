# -*- coding: utf-8 -*-
"""Backend FastAPI — expose les comparaisons du notebook + moteur de requête live."""
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = FastAPI(
    title="RAG Comparison API",
    description="Comparaison des méthodes de chunking, embedding et retrieval du pipeline RAG.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Résultat manquant : {name}. Exécutez d'abord le notebook.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- API résultats
@app.get("/api/health")
def health():
    return {"status": "ok", "results_present": sorted(p.name for p in RESULTS_DIR.glob("*.json"))}


@app.get("/api/results/{stage}")
def get_results(stage: str):
    if stage not in {"chunking", "embedding", "retrieval"}:
        raise HTTPException(status_code=404, detail=f"Étape inconnue : {stage}")
    return load_json(f"{stage}.json")


@app.get("/api/vectorial")
def get_vectorial():
    """Bouton « vectorisation » cité par le TP3 : alias de l'étape embedding."""
    return get_results("embedding")


@app.get("/api/best-methods")
def get_best_methods():
    return load_json("best_methods.json")


@app.get("/api/llm-benchmark")
def get_llm_benchmark():
    return load_json("llm_benchmark.json")


@app.get("/architecture.png")
def architecture_image():
    """Schéma d'architecture généré par le notebook (docs/architecture.png)."""
    path = BASE_DIR / "docs" / "architecture.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="architecture.png manquant — exécutez le notebook.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/architecture")
def get_architecture():
    """Métadonnées d'architecture pour la vue dédiée."""
    artifacts = load_json("artifacts.json")
    return {
        "offline": [
            "Cleaning du PDF (191 → ~100 pages) — notebook PARTIE 1",
            "Chunking : 7 méthodes comparées → gagnant",
            "Embedding : 7 méthodes comparées → gagnant",
            "Retrieval : 6 méthodes comparées → gagnant",
            "Export results/*.json (source de vérité unique)",
        ],
        "online": [
            "Frontend React + Recharts (ce site)",
            "Backend FastAPI : /api/results/{stage}, /api/best-methods, /api/query",
            "Moteur live : retrieval 6 méthodes + génération Ollama (repli Hugging Face)",
        ],
        "stack": {
            "notebook": "Python 3.13 · sentence-transformers · gensim · faiss-cpu · rank-bm25",
            "backend": "FastAPI + Uvicorn",
            "frontend": "React 18 + Vite + Recharts",
            "llm": artifacts.get("llm", {}),
        },
        "pipeline": artifacts.get("winners", {}),
        "corpus": {k: artifacts.get(k) for k in ("corpus_path", "corpus_chars", "n_chunks", "embedding_dim")},
    }


# ---------------------------------------------------------------- Moteur live
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    method: str | None = None  # None => méthode gagnante
    top_k: int = Field(4, ge=1, le=10)


class Engine:
    """Moteur live : recharge les artefacts du notebook (chunks + embeddings + BM25) une seule fois."""

    def __init__(self):
        with open(RESULTS_DIR / "embeddings_gagnant.pkl", "rb") as f:
            art = pickle.load(f)
        self.chunks: list[str] = art["chunks"]
        self.chunk_embs = np.asarray(art["chunk_embeddings"], dtype="float32")
        self.dpr_chunk_embs = np.asarray(art.get("dpr_chunk_embs", self.chunk_embs), dtype="float32")
        self.dpr_model_name = art.get("dpr_model")
        self.bm25 = art["bm25"]
        self.embedding_model_name = art["embedding_model"]
        self.cross_encoder_name = art.get("cross_encoder_model")
        self.winner = load_json("best_methods.json").get("retrieval", {}).get("name")
        self._st = None  # SentenceTransformer (chargé paresseusement)
        self._dpr = None
        self._ce = None

    def _embed_query(self, question: str) -> np.ndarray:
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.embedding_model_name)
        v = self._st.encode([question], normalize_embeddings=True)[0]
        return np.asarray(v, dtype="float32")

    def _embed_query_dpr(self, question: str) -> np.ndarray:
        if self._dpr is None:
            from sentence_transformers import SentenceTransformer
            self._dpr = SentenceTransformer(self.dpr_model_name or self.embedding_model_name)
        v = self._dpr.encode([question], normalize_embeddings=True)[0]
        return np.asarray(v, dtype="float32")

    # --- méthodes de retrieval live (retournent (liste d'indices, temps)) ---
    def retrieve(self, question: str, method: str, k: int) -> tuple[list[int], float]:
        from sklearn.metrics.pairwise import cosine_similarity
        t0 = time.perf_counter()
        if method in ("brute", "ann", "dense", "hybrid", "rerank") or method is None:
            q = self._embed_query(question)
            sims = cosine_similarity(q.reshape(1, -1), self.chunk_embs)[0]
            dense_idx = list(np.argsort(sims)[::-1][:max(k, 25)])
        if method == "brute" or method is None:
            idx = dense_idx[:k]
        elif method == "ann":
            import faiss
            if not hasattr(self, "_hnsw"):
                dim = self.chunk_embs.shape[1]
                self._hnsw = faiss.IndexHNSWFlat(dim, 32)
                self._hnsw.add(self.chunk_embs)
            _, idx_arr = self._hnsw.search(q.reshape(1, -1), k)
            idx = [int(i) for i in idx_arr[0]]
        elif method == "bm25":
            scores = self.bm25.get_scores(self._bm25_tokenize(question))
            idx = list(np.argsort(scores)[::-1][:k])
        elif method == "dense":
            qd = self._embed_query_dpr(question)
            sims_d = (qd.reshape(1, -1) @ self.dpr_chunk_embs.T)[0]
            idx = list(np.argsort(sims_d)[::-1][:k])
        elif method == "hybrid":
            rrf: dict[int, float] = {}
            for rank, i in enumerate(dense_idx[:25]):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
            bm_idx = list(np.argsort(self.bm25.get_scores(self._bm25_tokenize(question)))[::-1][:25])
            for rank, i in enumerate(bm_idx):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
            idx = sorted(rrf, key=rrf.get, reverse=True)[:k]
        elif method == "rerank":
            cand = dense_idx[:25]
            ce = self._cross_encoder()
            pairs = [(question, self.chunks[i][:512]) for i in cand]
            scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
            order = np.argsort(scores)[::-1]
            idx = [int(cand[j]) for j in order[:k]]
        else:
            raise HTTPException(status_code=400, detail=f"Méthode inconnue : {method}")
        return [int(i) for i in idx], time.perf_counter() - t0

    def _cross_encoder(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.cross_encoder_name)
        return self._ce

    @staticmethod
    def _bm25_tokenize(t: str) -> list[str]:
        return [w for w in re.findall(r"[a-zàâäéèêëîïôöùûüç]{2,}", t.lower())
                if w not in {"les", "des", "une", "dans", "pour", "avec", "sont", "cette",
                             "leur", "plus", "mais", "que", "qui", "par", "sur", "entre",
                             "ainsi", "cela", "afin", "donc"}]


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


METHOD_LABELS = {
    "brute": "1. Similarité brute",
    "ann": "2. ANN rapide (FAISS)",
    "bm25": "3. Mots-clés (BM25)",
    "dense": "4. Dense spécialisé (DPR)",
    "hybrid": "5. Hybride (BM25+dense)",
    "rerank": "6. Re-ranking (2 étapes)",
}
LABEL_TO_METHOD = {v: k for k, v in METHOD_LABELS.items()}


class QueryResponse(BaseModel):
    question: str
    method: str
    passages: list[dict]
    timings_ms: list[dict]
    generation: dict | None = None


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    engine = get_engine()
    requested = req.method
    if requested and requested in LABEL_TO_METHOD:
        requested = LABEL_TO_METHOD[requested]
    methods = list(METHOD_LABELS.keys()) if (requested in (None, "all")) else [requested]

    timings = []
    results_by_method: dict[str, list[int]] = {}
    for m in methods:
        idx, dt = engine.retrieve(req.question, m, req.top_k)
        results_by_method[m] = idx
        timings.append({"method": METHOD_LABELS[m], "ms": round(dt * 1000, 1)})

    winner_key = None
    if engine.winner:
        for k, v in METHOD_LABELS.items():
            if v == engine.winner:
                winner_key = k
    chosen = winner_key if winner_key in results_by_method else methods[0]
    idx = results_by_method[chosen]

    passages = [
        {
            "rank": i + 1,
            "index": int(idx[i]),
            "text": engine.chunks[idx[i]],
            "method": METHOD_LABELS[chosen],
        }
        for i in range(len(idx))
    ]

    generation = None
    try:
        from backend.llm_client import generate_with_context
        generation = generate_with_context(req.question, [p["text"] for p in passages])
    except Exception as e:  # la génération ne doit jamais casser le retrieval
        generation = {"backend": "indisponible", "answer": f"(génération indisponible : {e})", "sources": []}

    return QueryResponse(
        question=req.question,
        method=METHOD_LABELS[chosen],
        passages=passages,
        timings_ms=timings,
        generation=generation,
    )


# ---------------------------------------------------------------- Frontend statique
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:  # mode dev : le frontend tourne sous Vite (port 5173)
    @app.get("/")
    def root():
        return {"message": "API RAG OK. Frontend non buildé — lancez-le avec 'npm run dev' dans frontend/."}

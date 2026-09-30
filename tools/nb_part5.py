# -*- coding: utf-8 -*-
"""Partie 5 du notebook : architecture, artefacts pour l'app web, synthese."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code

cells = []

# ============ ARCHITECTURE ============
cells.append(md("""# PARTIE 6 — Architecture du projet

Le notebook produit les **données** ; l'application web les **expose** :

```
┌────────────────────────────  OFFLINE (ce notebook)  ────────────────────────────┐
│ thèse_pdf.pdf ─► CLEANING ─► CORPUS 100 p. ─► CHUNKING ×7 ─► gagnant            │
│                                                    gagnant ─► EMBEDDING ×7 ─► gagnant │
│                                                                   gagnant ─► RETRIEVAL ×6 ─► gagnant │
│                                                                                │
│ Exports : results/*.json (comparaisons + gagnants justifiés)                    │
│           + artefacts (chunks gagnants, embeddings, index BM25, modèles)        │
└──────────────────────────────────────│─────────────────────────────────────────┘
                                        ▼
┌────────────────────────────  ONLINE (application web)  ─────────────────────────┐
│  FRONTEND React (Vite, Recharts)  ──REST──►  BACKEND FastAPI   ──►  Ollama      │
│  · vue RAG Workflow (question → réponse)        · GET /api/results/{stage}      │
│  · vues comparatives (graphiques animés)        · GET /api/best-methods         │
│  · bannière "meilleure méthode" + justification │ · POST /api/query (live+LLM)  │
│  · console de question                          · sert le build frontend        │
└─────────────────────────────────────────────────────────────────────────────────┘
```
"""))

cells.append(code("""# ---------- Génération de l'image d'architecture (docs/architecture.png) ----------
# Pipeline LINÉAIRE : corpus -> chunking -> embedding -> retrieval (chaque gagnant nourrit l'étape suivante).
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(11.5, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, label, color, fs=9):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                         linewidth=1.4, edgecolor="#333", facecolor=color))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2, style="-|>", color="#333", lw=1.3, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls))

# ---- OFFLINE : flux linéaire d'évaluation ----
ax.text(0.3, 9.6, "OFFLINE — Notebook (source de vérité)", fontsize=11, fontweight="bold")

box(0.3, 8.5, 1.6, 0.9, "PDF brut\\n(thèse 191 p.)", "#e07a5f")
box(2.3, 8.5, 1.8, 0.9, "CLEANING\\n→ corpus 100 p.", "#f2cc8f")
box(4.5, 8.5, 1.8, 0.9, "CORPUS", "#f2cc8f")
arrow(1.9, 8.95, 2.3, 8.95)
arrow(4.1, 8.95, 4.5, 8.95)

box(1.2, 6.7, 2.0, 1.0, "CHUNKING\\n7 méthodes → gagnant", "#81b29a")
box(3.7, 6.7, 2.0, 1.0, "EMBEDDING\\n7 méthodes → gagnant", "#81b29a")
box(6.2, 6.7, 2.0, 1.0, "RETRIEVAL\\n6 méthodes → gagnant", "#81b29a")
arrow(4.9, 8.5, 2.2, 7.7)   # corpus -> chunking
arrow(3.2, 7.2, 3.7, 7.2)   # chunking -> embedding (chunks gagnants)
arrow(5.7, 7.2, 6.2, 7.2)   # embedding -> retrieval (embeddings gagnants)

box(8.7, 6.7, 1.1, 1.0, "results/\\n*.json", "#cb997e")
arrow(8.2, 7.2, 8.7, 7.2)   # retrieval -> exports JSON

# ---- ONLINE ----
ax.text(0.3, 5.1, "ONLINE — Application web (FastAPI + React)", fontsize=11, fontweight="bold")

box(0.3, 2.4, 3.4, 1.6, "FRONTEND React + Recharts\\nvues comparatives, RAG Workflow,\\nconsole de question", "#a8dadc")
box(4.6, 2.4, 3.3, 1.6, "BACKEND FastAPI\\n/api/results · /api/best-methods\\n/api/query (live + LLM)", "#89b0ae")
box(8.5, 2.4, 1.2, 1.6, "Ollama\\n(llama3.2,\\nnomic-embed)", "#bee9e8")
arrow(3.7, 3.2, 4.6, 3.2, style="<->")
arrow(7.9, 3.2, 8.5, 3.2, style="<->")

box(4.6, 0.6, 3.3, 0.9, "Notebook → artefacts (chunks, modèles)", "#ffe8d6")
arrow(6.25, 1.5, 6.25, 2.4)              # artefacts -> backend (moteur live)
arrow(9.2, 6.7, 7.0, 4.0, ls=":", color="#888")  # exports JSON -> backend (vues comparatives)

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=140, bbox_inches="tight")
plt.show()
print("docs/architecture.png généré.")
"""))

# ============ ARTEFACTS POUR L'APP ============
cells.append(md("""# PARTIE 7 — Artefacts pour l'application web

Le backend FastAPI ne recalcule **rien** : il recharge les artefacts exportés ici (chunks gagnants, embeddings gagnants, index BM25/FAISS, modèles) pour offrir la console de question en live, et lit les JSON de comparaison pour les vues graphiques.
"""))

cells.append(code("""# ---------- Export des artefacts du pipeline gagnant ----------
import pickle

artifacts = {
    "corpus_path": "data/CORPUS_RAG_reduit_100pages.txt",
    "corpus_chars": len(corpus_text),
    "cleaning": {k: v for k, v in info.items() if k != "text"},
    "winners": {
        "chunking": best_chunking_name,
        "embedding": best_embedding_name,
        "retrieval": best_retrieval_name,
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "dpr_model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "n_chunks": len(best_chunks),
    "embedding_dim": int(chunk_embs.shape[1]),
    "llm": {"ollama_ready": bool(ollama_ready), "model": GEN_MODEL,
            "benchmark": llm_rows},
    "test_queries": test_queries,
}
with open("results/artifacts.json", "w", encoding="utf-8") as f:
    json.dump(artifacts, f, ensure_ascii=False, indent=2)

with open("results/chunks.json", "w", encoding="utf-8") as f:
    json.dump({"chunking_method": best_chunking_name, "chunks": best_chunks},
              f, ensure_ascii=False)

# Embeddings + BM25 + DPR : pickles (rechargés par le backend)
import faiss as _faiss
with open("results/embeddings_gagnant.pkl", "wb") as f:
    pickle.dump({"chunks": best_chunks,
                 "chunk_embeddings": chunk_embs,
                 "dpr_chunk_embs": dpr_chunk_embs,
                 "bm25": bm25,
                 "embedding_model": artifacts["embedding_model"],
                 "dpr_model": artifacts["dpr_model"],
                 "cross_encoder_model": artifacts["cross_encoder_model"]}, f)
print("results/artifacts.json, chunks.json, embeddings_gagnant.pkl écrits.")
"""))

# ============ SYNTHESE ============
cells.append(md("""# Synthèse finale

| Étape | Méthode gagnante | Pourquoi (résumé) |
|---|---|---|
| **Cleaning** | nettoyage automatique par détection de sections | 191 → 100 pages, retraits exacts demandés par l'encadrante |
| **Chunking** | voir `results/best_methods.json` | meilleur couple cohérence interne / redondance / F1 |
| **Embedding** | voir `results/best_methods.json` | espace pré-entraîné : meilleur MRR/nDCG sur les pseudo-questions |
| **Retrieval** | voir `results/best_methods.json` | meilleur compromis qualité de classement / latence |
| **Génération** | Ollama (repli HF) | réponses citées, 100 % local, latence mesurée par modèle |

**Livrables du projet** :
1. ce notebook exécuté de bout en bout (nettoyage → 3 comparaisons → génération) ;
2. `results/*.json` + `docs/*.png` pour le mémoire ;
3. l'application web FastAPI + React qui présente ces comparaisons avec la meilleure méthode **désignée et justifiée** à chaque étape, et la console de question branchée sur le pipeline gagnant.
"""))

cells.append(code("""print("=== RÉCAPITULATIF DES EXPORTS ===")
import os
for f in sorted(os.listdir("results")):
    size = os.path.getsize(os.path.join("results", f))
    print(f"  results/{f:28s} {size/1024:8.1f} Ko")
for f in sorted(os.listdir("docs")):
    size = os.path.getsize(os.path.join("docs", f))
    print(f"  docs/{f:32s} {size/1024:8.1f} Ko")
print("\\nNotebook terminé — l'application web peut être lancée : uvicorn backend.main:app --port 8001")
"""))

part5 = cells

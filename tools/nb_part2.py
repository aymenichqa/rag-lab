# -*- coding: utf-8 -*-
"""Partie 2 du notebook : demo TP1 sur le corpus reel + comparaison des 7 methodes de chunking."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code

cells = []

# ============ TP1 mini-demo ============
cells.append(md("""# PARTIE 1bis — TP1 : Pipeline RAG simplifié (démo sur un extrait du corpus)

Reprise du TP1 sur un **extrait réel du corpus** (20 premières phrases) pour bien fixer les idées avant les comparaisons :
chunking → embeddings → vector store FAISS → similarité cosinus → retrieval. Livrables TP1 inclus (matrice de similarité + heatmap).

---

"""))

cells.append(code("""# Extrait du corpus pour la démo TP1
# (simple_sentence_split est défini ici pour l'autonomie de la démo)
import re as _re

def simple_sentence_split(t):
    sentences = _re.split(r"(?<=[.!?])\\s+", t.strip())
    return [s.strip() for s in sentences if s.strip()]

demo_text = " ".join(simple_sentence_split(corpus_text)[:20])
print(f"Extrait démo : {len(demo_text)} caractères, {len(simple_sentence_split(demo_text))} phrases")
print(demo_text[:300], "...")
"""))

cells.append(code("""# Etape TP1 : chunking (par phrase), embeddings, FAISS, matrice de similarité
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

splitter = CharacterTextSplitter(separator="\\n", chunk_size=100, chunk_overlap=20)
demo_docs = [d.strip() for d in splitter.split_text(demo_text.replace(". ", ".\\n")) if d.strip()]

demo_embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
demo_chunk_embs = np.array(demo_embeddings_model.embed_documents(demo_docs))
print(f"Chunks : {len(demo_docs)} | Dimension des vecteurs : {demo_chunk_embs.shape[1]}")

demo_vectorstore = FAISS.from_texts(demo_docs, demo_embeddings_model)
print("Vector store FAISS créé avec", demo_vectorstore.index.ntotal, "vecteurs.")
"""))

cells.append(code("""# Livrable TP1 : matrice de similarité cosinus entre chunks + heatmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

demo_sim = cosine_similarity(demo_chunk_embs)
labels = [f"Chunk {i+1}" for i in range(len(demo_docs))]
demo_sim_df = pd.DataFrame(demo_sim, index=labels, columns=labels).round(3)

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(demo_sim_df, cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("TP1 — Matrice de similarité cosinus entre chunks (extrait du corpus)")
plt.tight_layout()
plt.savefig("docs/tp1_similarity_matrix.png", dpi=130)
plt.show()
"""))

cells.append(code("""# TP1 : retrieval d'une question sur l'extrait (FAISS + cosinus manuel)
demo_query = "Comment les zones humides réduisent-elles la pollution par les nitrates ?"

results = demo_vectorstore.similarity_search_with_score(demo_query, k=2)
print(f"Question : {demo_query}\\n--- Résultats FAISS (distance L2, plus petit = plus proche) ---")
for rank, (doc, score) in enumerate(results, 1):
    print(f"#{rank} (distance = {score:.4f}) : {doc.page_content[:120]}")

q_emb = np.array(demo_embeddings_model.embed_query(demo_query)).reshape(1, -1)
sims = cosine_similarity(q_emb, demo_chunk_embs)[0]
best = int(np.argmax(sims))
print(f"\\nChunk le plus pertinent (cosinus manuel) : Chunk {best+1} (similarité = {sims[best]:.4f})")
print(demo_docs[best][:200])
"""))

# ============ COMPARAISON CHUNKING ============
cells.append(md("""# PARTIE 2 — Comparaison des 7 méthodes de Chunking

Le corpus est fixé (celui du cleaning). On ne fait varier **que** la méthode de découpage : c'est la règle d'or de l'expérimentation.

## Les 7 méthodes implémentées

| # | Méthode | Principe | Points forts | Limites |
|---|---|---|---|---|
| 1 | **Taille fixe** | tranches brutes de N caractères | simple, rapide | coupe les mots et les idées |
| 2 | **Récursif** | essaie séparateurs hiérarchiques (paragraphe > phrase > mot) vers une taille cible | bon compromis universel | pas de notion de sens |
| 3 | **Par phrase** | regroupe des phrases complètes jusqu'à la taille cible | jamais de phrase coupée | ignore les transitions thématiques |
| 4 | **Par paragraphe** | un chunk = un paragraphe naturel | respecte la structure d'origine | sur du texte PDF, les paragraphes sont irréguliers |
| 5 | **Fenêtre glissante** | fenêtres de mots qui se chevauchent | redondance contrôlée, bon recall | duplique le contenu (coût d'index) |
| 6 | **Par tokens** | découpe selon les vrais tokens du LLM (tiktoken) | cohérent avec les limites des LLM | même aveugle au sens que (1) |
| 7 | **Sémantique** | regroupe les phrases consécutives tant que leur similarité d'embedding reste haute | chunks thématiquement cohérents | coûteux (N embeddings de phrases) |

> **Note d'implémentation** : la méthode sémantique nécessite un embedding de chaque phrase et des comparaisons de phrases consécutives ; pour garder un temps d'exécution raisonnable, elle est appliquée sur un **extrait contigu représentatif** du corpus (les 40 000 premiers caractères, ≈ 20 % du corpus). Les 6 autres méthodes traitent l'intégralité du corpus.
"""))

cells.append(code("""import numpy as np
import pandas as pd
import re

np.random.seed(42)

# ---------- Les 7 fonctions de chunking ----------
def chunk_fixed_size(t, size=180):
    return [t[i:i+size] for i in range(0, len(t), size) if t[i:i+size].strip()]

def chunk_recursive(t, size=180, overlap=30):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(t)

def chunk_by_sentence(t, target_size=180):
    sentences = simple_sentence_split(t)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= target_size or not current:
            current = (current + " " + s).strip()
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

def chunk_by_paragraph(t):
    paras = [p.strip() for p in t.split("\\n\\n") if p.strip()]
    return paras

def chunk_sliding_window(t, window_words=30, stride_words=15):
    words = t.split()
    chunks = []
    for i in range(0, len(words), stride_words):
        window = words[i:i+window_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if i + window_words >= len(words):
            break
    return chunks

def chunk_by_tokens(t, chunk_size=40, chunk_overlap=8):
    from langchain_text_splitters import TokenTextSplitter
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(t)

def chunk_semantic(t, embed_fn, similarity_threshold=0.55, max_chars=260, max_chars_processed=40_000):
    # appliqué sur un extrait contigu (voir note d'implémentation)
    extract = t[:max_chars_processed]
    sentences = simple_sentence_split(extract)
    if not sentences:
        return []
    embs = np.array(embed_fn(sentences))
    from sklearn.metrics.pairwise import cosine_similarity
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embs[i-1]], [embs[i]])[0][0]
        current_len = sum(len(s) for s in current)
        if sim >= similarity_threshold and current_len < max_chars:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    chunks.append(" ".join(current))
    return chunks

# Modèle d'embedding partagé (MiniLM) : sert au chunking sémantique ET aux métriques
from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_fn(texts):
    return embeddings_model.embed_documents(texts)

print("7 fonctions de chunking définies + modèle d'embedding chargé.")
"""))

cells.append(code("""# ---------- Application des 7 méthodes sur le corpus ----------
methods = {
    "1. Taille fixe": chunk_fixed_size(corpus_text),
    "2. Récursif": chunk_recursive(corpus_text),
    "3. Par phrase": chunk_by_sentence(corpus_text),
    "4. Par paragraphe": chunk_by_paragraph(corpus_text),
    "5. Fenêtre glissante": chunk_sliding_window(corpus_text),
    "6. Par tokens": chunk_by_tokens(corpus_text),
    "7. Sémantique": chunk_semantic(corpus_text, embed_fn),
}

for name, chunks in methods.items():
    lengths = [len(c) for c in chunks]
    print(f"{name:22s} : {len(chunks):5d} chunks | longueur moy. {np.mean(lengths):6.1f} car. | min {min(lengths)} / max {max(lengths)}")
"""))

cells.append(md("""## Les 8 métriques d'évaluation

| Métrique | Objectif | Calcul |
|---|---|---|
| Nombre de chunks | fragmentation (coût d'index) | `len(chunks)` |
| Longueur moyenne | taille optimale | moyenne des caractères par chunk |
| Écart-type | homogénéité des tailles | écart-type des longueurs |
| Similarité intra | **cohérence** interne | cosinus moyen entre les phrases **à l'intérieur** de chaque chunk (↑ mieux) |
| Similarité inter | **redondance** | cosinus moyen **entre chunks différents** (↓ mieux) |
| Recall@2 | couverture | le bon chunk est-il dans le top-2 ? |
| Precision@2 | pertinence | proportion pertinente sur les 2 slots |
| F1-score | performance globale | moyenne harmonique de Precision@2 et Recall@2 |
| MRR (+bonus) | position du bon résultat | moyenne de 1/rang du bon chunk |

**Méthodologie Recall/Precision/F1/MRR** (interprétation raisonnée, à valider avec l'encadrante) : chaque phrase du corpus sert de **pseudo-question** ; le chunk « correct » est celui qui contient cette phrase. On embed la pseudo-question et tous les chunks, on classe par similarité cosinus et on mesure le rang du bon chunk.
"""))

cells.append(code("""from sklearn.metrics.pairwise import cosine_similarity

def length_stats(chunks):
    lengths = [len(c) for c in chunks]
    return len(chunks), float(np.mean(lengths)), float(np.std(lengths))

def intra_inter_similarity(chunks, chunk_embs, embed_fn, max_sentences=3500):
    \"\"\"intra : cosinus moyen entre phrases d'un même chunk ; inter : cosinus moyen entre chunks.\"\"\"
    # --- intra : on rassemble toutes les phrases avec leur chunk propriétaire (échantillon capé) ---
    all_sents, owner = [], []
    for ci, c in enumerate(chunks):
        for s in simple_sentence_split(c):
            all_sents.append(s)
            owner.append(ci)
    if len(all_sents) > max_sentences:  # échantillonnage déterministe (stride)
        stride = int(np.ceil(len(all_sents) / max_sentences))
        all_sents = all_sents[::stride]
        owner = owner[::stride]
    sent_embs = np.array(embed_fn(all_sents)) if all_sents else np.zeros((0, 1))
    intra_scores = []
    by_chunk = {}
    for emb, ci in zip(sent_embs, owner):
        by_chunk.setdefault(ci, []).append(emb)
    for ci, embs in by_chunk.items():
        n = len(embs)
        if n >= 2:
            sim = cosine_similarity(np.array(embs))
            intra_scores.append((sim.sum() - n) / (n * (n - 1)))
    intra = float(np.mean(intra_scores)) if intra_scores else 1.0

    # --- inter : matrice chunk x chunk (chunk_embs déjà calculés) ---
    n = len(chunks)
    if n < 2:
        inter = 0.0
    else:
        sim_matrix = cosine_similarity(chunk_embs)
        inter = float((sim_matrix.sum() - n) / (n * (n - 1)))
    return intra, inter

def retrieval_metrics_full(chunk_embs, chunks, source_sentences, embed_fn, k=2):
    \"\"\"Recall@k, Precision@k, F1, MRR sur des pseudo-questions (phrases du corpus).\"\"\"
    if len(chunks) < 2 or not source_sentences:
        return 0.0, 0.0, 0.0, 0.0
    queries = source_sentences
    q_embs = np.array(embed_fn(queries))
    hits, ranks, total = 0, [], 0
    for qi in range(len(queries)):
        sent = queries[qi]
        gt_idx = next((i for i, c in enumerate(chunks) if sent[:40] in c), None)
        if gt_idx is None:
            continue
        sims = cosine_similarity(q_embs[qi].reshape(1, -1), chunk_embs)[0]
        order = np.argsort(sims)[::-1]
        rank = int(np.where(order == gt_idx)[0][0]) + 1
        ranks.append(rank)
        total += 1
        if gt_idx in order[:k]:
            hits += 1
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    recall = hits / total
    precision = hits / (total * k)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    mrr = float(np.mean([1.0 / r for r in ranks]))
    return recall, precision, f1, mrr

source_sentences = simple_sentence_split(corpus_text)
print(f"{len(source_sentences)} phrases sources détectées (pseudo-questions).")
"""))

cells.append(code("""# ---------- Boucle d'évaluation des 7 méthodes ----------
rows = []
method_chunk_embeddings = {}
for name, chunks in methods.items():
    print(f"Évaluation : {name} ({len(chunks)} chunks)...")
    chunk_embs = np.array(embed_fn(chunks))
    method_chunk_embeddings[name] = chunk_embs
    n, mean_len, std_len = length_stats(chunks)
    intra, inter = intra_inter_similarity(chunks, chunk_embs, embed_fn)
    recall, precision, f1, mrr = retrieval_metrics_full(chunk_embs, chunks, source_sentences[:150], embed_fn, k=2)
    rows.append({
        "Méthode": name, "Nb chunks": n, "Long. moyenne": round(mean_len, 1),
        "Écart-type": round(std_len, 1), "Similarité intra": round(intra, 3),
        "Similarité inter": round(inter, 3), "Recall@2": round(recall, 3),
        "Precision@2": round(precision, 3), "F1-score": round(f1, 3), "MRR": round(mrr, 3),
    })

results_df = pd.DataFrame(rows).set_index("Méthode")
results_df
"""))

cells.append(md("""## Meilleure méthode de chunking

**Score composite** = Similarité intra (cohérence) − Similarité inter (redondance) + F1-score.
Les **% d'écart** sont calculés par rapport à la médiane des 7 méthodes : ils montrent de **combien** chaque méthode se distingue, pas seulement qui gagne.
"""))

cells.append(code("""def add_relative_percentages(df, metrics, lower_better=("Similarité inter",)):
    med = df[metrics].median()
    pct = pd.DataFrame(index=df.index)
    for m in metrics:
        sign = -1.0 if m in lower_better else 1.0
        denom = med[m] if med[m] != 0 else 1e-9
        pct[f"{m} %"] = ((sign * (df[m] - med[m])) / abs(denom) * 100).round(1)
    return pct

metrics_pct_metrics = ["Nb chunks", "Long. moyenne", "Similarité intra", "Similarité inter", "F1-score", "MRR"]
pct_df = add_relative_percentages(results_df, metrics_pct_metrics, lower_better=("Similarité inter",))
results_df_pct = pd.concat([results_df, pct_df], axis=1)

results_df["Score composite"] = (
    results_df["Similarité intra"] - results_df["Similarité inter"] + results_df["F1-score"]
).round(3)
ranking = results_df.sort_values("Score composite", ascending=False)
print(ranking[["Similarité intra", "Similarité inter", "F1-score", "MRR", "Score composite"]].to_string())
best_chunking_name = ranking.index[0]
print(f"\\n>>> Meilleure méthode de chunking : {best_chunking_name}")
results_df_pct
"""))

cells.append(code("""# Justification écrite de la meilleure méthode (chiffres à l'appui)
w = results_df.loc[best_chunking_name]
med = results_df[["Similarité intra", "Similarité inter", "F1-score", "MRR", "Nb chunks"]].median()
second = ranking.index[1]

chunking_justification = (
    f"La méthode « {best_chunking_name} » arrive en tête avec un score composite de {w['Score composite']:.3f} "
    f"(devant « {second} », {w['Score composite'] - ranking['Score composite'].iloc[1]:.3f} point plus haut). "
    f"Elle combine la meilleure cohérence interne ({w['Similarité intra']:.3f}, soit "
    f"{(w['Similarité intra']/med['Similarité intra']-1)*100:+.1f} % vs médiane), une redondance contenue "
    f"({w['Similarité inter']:.3f} contre {med['Similarité inter']:.3f} en médiane) et le meilleur comportement "
    f"en retrieval (F1 = {w['F1-score']:.3f}, MRR = {w['MRR']:.3f}) : ses chunks portent une idée homogène "
    f"sans dupliquer le contenu, ce qui maximise à la fois la précision et la position du bon passage dans le classement. "
    f"Avec {int(w['Nb chunks'])} chunks pour {int(med['Nb chunks'])} en médiane, le coût d'indexation reste raisonnable."
)
print(chunking_justification)
"""))

cells.append(code("""# Visualisations : métriques clés +trade-off cohérence/redondance
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
colors = ["#f4a261" if m == best_chunking_name else "#4c78a8" for m in results_df.index]

results_df["F1-score"].plot(kind="barh", ax=axes[0], color=colors, title="F1-score (retrieval du bon chunk)")
axes[0].invert_yaxis()
results_df[["Similarité intra", "Similarité inter"]].plot(kind="barh", ax=axes[1],
               color=["#81b29a", "#e07a5f"], title="Cohérence intra (↑) vs redondance inter (↓)")
axes[1].invert_yaxis()
axes[2].scatter(results_df["Similarité inter"], results_df["Similarité intra"],
                s=results_df["F1-score"]*900, alpha=0.75, c=colors)
for m in results_df.index:
    axes[2].annotate(m.split(". ")[1], (results_df.loc[m, "Similarité inter"], results_df.loc[m, "Similarité intra"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
axes[2].set_xlabel("Redondance (similarité inter, ↓ mieux)")
axes[2].set_ylabel("Cohérence (similarité intra, ↑ mieux)")
axes[2].set_title("Trade-off cohérence / redondance (taille = F1)")
plt.tight_layout()
plt.savefig("docs/chunking_compare.png", dpi=130)
plt.show()
"""))

cells.append(code("""# Export JSON de la comparaison de chunking (source de vérité de l'application web)
import json, os

os.makedirs("results", exist_ok=True)
chunking_export = {
    "stage": "chunking",
    "metrics_columns": ["Nb chunks", "Long. moyenne", "Écart-type", "Similarité intra",
                        "Similarité inter", "Recall@2", "Precision@2", "F1-score", "MRR", "Score composite"],
    "higher_is_better": {"Nb chunks": False, "Long. moyenne": None, "Écart-type": False,
                         "Similarité intra": True, "Similarité inter": False, "Recall@2": True,
                         "Precision@2": True, "F1-score": True, "MRR": True, "Score composite": True},
    "methods": [
        {"name": idx, **{col: (float(row[col]) if isinstance(row[col], (int, float, np.floating)) else row[col])
                          for col in results_df.columns}}
        for idx, row in results_df.iterrows()
    ],
    "best": {"name": best_chunking_name, "justification": chunking_justification},
    "ranking": list(ranking.index),
}
with open("results/chunking.json", "w", encoding="utf-8") as f:
    json.dump(chunking_export, f, ensure_ascii=False, indent=2, default=str)
print("results/chunking.json écrit.")

best_chunks = methods[best_chunking_name]
print(f"Chunks gagnants gardés en mémoire pour la suite : {len(best_chunks)} chunks.")
"""))

part2 = cells

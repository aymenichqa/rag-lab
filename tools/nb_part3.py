# -*- coding: utf-8 -*-
"""Partie 3 du notebook : comparaison des 7 methodes d'embedding."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code

cells = []

# ============ COMPARAISON EMBEDDING ============
cells.append(md("""# PARTIE 3 — Comparaison des 7 méthodes d'Embedding

**Point de méthodologie** : le chunking est désormais **fixé** au gagnant de la Partie 2. On ne fait varier **que** la méthode de vectorisation.

## Les 7 méthodes

| # | Méthode | Principe | Entraînement | Points forts | Limites |
|---|---|---|---|---|---|
| 1 | **Word2Vec** | un vecteur par mot, agrégé par moyenne | sur le corpus local | léger, rapide, sans internet | n'a jamais vu le mot hors corpus ; perd l'ordre des mots |
| 2 | **GloVe** | co-occurrences globales, moyenne par chunk | pré-entraîné (Wikipedia) | vocabulaire mondial | moyenne de mots : le sens d'une phrase n'est pas un mot |
| 3 | **FastText** | mots + sous-mots (n-grammes de caractères) | sur le corpus local | gère les mots inconnus (OOV) via les sous-mots | même agrégation naïve que Word2Vec |
| 4 | **BERT (mean pooling)** | vecteurs **contextuels** (le même mot ≠ même vecteur selon le contexte) | pré-entraîné | capture le contexte local | brut, non optimisé pour la similarité de phrases |
| 5 | **Sentence-BERT (MiniLM)** | BERT re-entraîné pour que « phrases proches en sens = vecteurs proches » | pré-entraîné | LA méthode conçue pour le retrieval sémantique | — |
| 6 | **Doc2Vec** | un vecteur appris par document/chunk | sur le corpus local | vecteur de document natif | `infer_vector` non déterministe ; petit corpus |
| 7 | **Ollama (nomic-embed-text)** | modèle d'embedding moderne servi localement | pré-entraîné, local via API | pipeline 100 % local (sujet : Ollama) | nécessite Ollama actif |

**Hypothèse à vérifier** : les méthodes entraînées from scratch sur notre petit corpus (Word2Vec, FastText, Doc2Vec) devraient être nettement dominées par les pré-entraînées (GloVe, BERT, SBERT, nomic). Le comparatif sert justement à le démontrer chiffres à l'appui.
"""))

cells.append(code("""import glob
pdf_paths = sorted(glob.glob("*.pdf"))
print(f"Corpus fixé : {len(best_chunks)} chunks ({best_chunking_name}) — {len(corpus_text)} caractères")
print(f"PDF source présent : {pdf_paths}")
"""))

cells.append(code("""# ---------- 1. Word2Vec (entraîné sur le corpus) ----------
from gensim.models import Word2Vec

tokenized = [c.lower().split() for c in best_chunks]
w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=1, epochs=50)

def embed_word2vec(texts):
    out = []
    for t in texts:
        words = [w for w in t.lower().split() if w in w2v_model.wv]
        out.append(np.mean([w2v_model.wv[w] for w in words], axis=0) if words else np.zeros(100))
    return out
print("Word2Vec entraîné (corpus).")
"""))

cells.append(code("""# ---------- 2. GloVe (pré-entraîné, téléchargement ~130 Mo au 1er lancement) ----------
import gensim.downloader as gensim_api

glove_model = gensim_api.load("glove-wiki-gigaword-100")

def embed_glove(texts):
    out = []
    for t in texts:
        words = [w for w in t.lower().split() if w in glove_model]
        out.append(np.mean([glove_model[w] for w in words], axis=0) if words else np.zeros(100))
    return out
print("GloVe pré-entraîné chargé.")
"""))

cells.append(code("""# ---------- 3. FastText (entraîné sur le corpus) ----------
from gensim.models import FastText

ft_model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=1, epochs=50)

def embed_fasttext(texts):
    return [ft_model.wv.get_sentence_vector(t.lower().split()) for t in texts]
print("FastText entraîné (corpus).")
"""))

cells.append(code("""# ---------- 4. BERT (mean pooling, pré-entraîné) ----------
import torch
from transformers import AutoTokenizer, AutoModel

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def embed_bert(texts, batch_size=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            outputs = bert_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            mean_pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            out.extend(mean_pooled.numpy())
    return out
print("BERT chargé (mean pooling, par batch).")
"""))

cells.append(code("""# ---------- 5. Sentence-BERT (MiniLM, pré-entraîné) ----------
from langchain_huggingface import HuggingFaceEmbeddings

sbert_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_sbert(texts):
    return sbert_model.embed_documents(texts)
print("Sentence-BERT (MiniLM) chargé.")
"""))

cells.append(code("""# ---------- 6. Doc2Vec (entraîné sur le corpus) ----------
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

tagged_docs = [TaggedDocument(words=tok, tags=[str(i)]) for i, tok in enumerate(tokenized)]
d2v_model = Doc2Vec(tagged_docs, vector_size=100, window=5, min_count=1, workers=1, epochs=50)

def embed_doc2vec(texts):
    return [d2v_model.infer_vector(t.lower().split(), epochs=30) for t in texts]
print("Doc2Vec entraîné (corpus).")
"""))

cells.append(md("""## Les 8 métriques

| Métrique | Objectif | Calcul |
|---|---|---|
| Similarité intra | **cohérence** thématique des chunks selon cet espace | cosinus moyen entre phrases d'un même chunk (↑ mieux) |
| Similarité inter | **séparation** entre chunks (↓ mieux) | cosinus moyen entre chunks différents |
| Precision@2 | pertinence des top-2 | hits / (2 × nb requêtes) |
| Recall@2 | couverture | le bon chunk est-il dans le top-2 ? |
| MAP | performance globale de classement | moyenne de 1/rang du bon chunk |
| MRR | position du bon résultat | identique à MAP ici : **une seule** réponse pertinente par requête → MAP = MRR (résultat théorique, pas une erreur) |
| nDCG | qualité du classement | 1/log₂(rang+1), pénalise un bon résultat trouvé tard |
| Cosine sim. (moy) | similarité brute question↔bon chunk | cosinus moyen sur les paires (question, bon chunk) |

**Même méthodologie que la Partie 2** : phrases = pseudo-questions, bon chunk = celui qui contient la phrase. Ici, la même famille de chunks est embeddée par **7 espaces vectoriels différents** : seuls les embeddings changent.
"""))

cells.append(code("""from sklearn.metrics.pairwise import cosine_similarity

def intra_inter_for_embed(chunks, embed_fn):
    \"\"\"intra : cosinus moyen entre phrases d'un même chunk (échantillon capé) ; inter : cosinus moyen entre chunks.\"\"\"
    all_sents, owner = [], []
    for ci, c in enumerate(chunks):
        for s in simple_sentence_split(c):
            all_sents.append(s)
            owner.append(ci)
    if len(all_sents) > 2500:
        stride = int(np.ceil(len(all_sents) / 2500))
        all_sents, owner = all_sents[::stride], owner[::stride]
    sent_embs = np.array(embed_fn(all_sents)) if all_sents else np.zeros((0, 1))
    by_chunk = {}
    for emb, ci in zip(sent_embs, owner):
        by_chunk.setdefault(ci, []).append(emb)
    intra_scores = []
    for ci, embs in by_chunk.items():
        n = len(embs)
        if n >= 2:
            sim = cosine_similarity(np.array(embs))
            intra_scores.append((sim.sum() - n) / (n * (n - 1)))
    intra = float(np.mean(intra_scores)) if intra_scores else 1.0
    chunk_embs = np.array(embed_fn(chunks))
    n = len(chunks)
    if n < 2:
        inter = 0.0
    else:
        sim_matrix = cosine_similarity(chunk_embs)
        inter = float((sim_matrix.sum() - n) / (n * (n - 1)))
    return intra, inter, chunk_embs

def rank_metrics_for_embed(chunks, chunk_embs, embed_fn, source_sentences, k=2):
    \"\"\"Precision@k, Recall@k, MAP, MRR, nDCG, cosinus moyen sur pseudo-questions.\"\"\"
    if len(chunks) < 2 or not source_sentences:
        return dict(precision=0, recall=0, map_=0, mrr=0, ndcg=0, avg_cos=0)
    queries = source_sentences
    q_embs = np.array(embed_fn(queries))
    hits, ranks, cos_scores, total = 0, [], [], 0
    for qi in range(len(queries)):
        sent = queries[qi]
        gt_idx = next((i for i, c in enumerate(chunks) if sent[:40] in c), None)
        if gt_idx is None:
            continue
        sims = cosine_similarity(q_embs[qi].reshape(1, -1), chunk_embs)[0]
        order = np.argsort(sims)[::-1]
        rank = int(np.where(order == gt_idx)[0][0]) + 1
        ranks.append(rank)
        cos_scores.append(float(sims[gt_idx]))
        total += 1
        if gt_idx in order[:k]:
            hits += 1
    if total == 0:
        return dict(precision=0, recall=0, map_=0, mrr=0, ndcg=0, avg_cos=0)
    precision = hits / (total * k)
    recall = hits / total
    map_mrr = float(np.mean([1 / r for r in ranks]))
    ndcg = float(np.mean([1 / np.log2(r + 1) for r in ranks]))
    return dict(precision=round(precision, 3), recall=round(recall, 3), map_=round(map_mrr, 3),
                mrr=round(map_mrr, 3), ndcg=round(ndcg, 3), avg_cos=round(float(np.mean(cos_scores)), 3))

print("Fonctions de métriques embedding prêtes.")
"""))

cells.append(code("""# ---------- 7. Ollama (nomic-embed-text, modèle local) ----------
import requests

_ollama_session = requests.Session()  # keep-alive pour accélérer les appels

def embed_ollama(texts, model="nomic-embed-text", url="http://localhost:11434/api/embed", batch_size=32):
    # Embeddings via l'API /api/embed (batchée) d'Ollama — rapide même sur des milliers de textes.
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        r = _ollama_session.post(url, json={"model": model, "input": batch}, timeout=120)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return out

try:
    _probe = embed_ollama(["test de disponibilité"])
    ollama_available = True
    print(f"Ollama disponible ({len(_probe[0])} dimensions).")
except Exception as e:
    ollama_available = False
    print(f"Ollama indisponible ({e}) -> la méthode sera notée N/A dans le tableau.")
"""))

cells.append(code("""# ---------- Évaluation des 7 méthodes d'embedding ----------
embedding_methods = {
    "Word2Vec": embed_word2vec,
    "GloVe": embed_glove,
    "FastText": embed_fasttext,
    "BERT": embed_bert,
    "Sentence-BERT": embed_sbert,
    "Doc2Vec": embed_doc2vec,
}
if ollama_available:
    embedding_methods["Ollama"] = embed_ollama

emb_rows = []
embed_chunk_embs = {}
for name, fn in embedding_methods.items():
    print(f"Évaluation : {name}...")
    try:
        intra, inter, cembs = intra_inter_for_embed(best_chunks, fn)
        embed_chunk_embs[name] = cembs
        m = rank_metrics_for_embed(best_chunks, cembs, fn, source_sentences[:150], k=2)
        emb_rows.append({
            "Méthode": name, "Cosine sim. (moy)": m["avg_cos"], "Sim. intra": round(intra, 3),
            "Sim. inter": round(inter, 3), "Precision@2": m["precision"], "Recall@2": m["recall"],
            "MAP": m["map_"], "MRR": m["mrr"], "NDCG": m["ndcg"],
        })
    except Exception as e:
        print(f"  ! {name} indisponible : {type(e).__name__} {str(e)[:100]}")
        emb_rows.append({"Méthode": name, "Cosine sim. (moy)": None, "Sim. intra": None,
                          "Sim. inter": None, "Precision@2": None, "Recall@2": None,
                          "MAP": None, "MRR": None, "NDCG": None})

import pandas as pd
emb_results_df = pd.DataFrame(emb_rows).set_index("Méthode")
emb_results_df
"""))

cells.append(md("""## Meilleure méthode d'embedding

**Score composite** = Sim. intra − Sim. inter + MRR + NDCG (cohérence, séparation, classement). Les % d'écart sont relatifs à la médiane.
"""))

cells.append(code("""valid = emb_results_df.dropna(subset=["MRR"]).copy()
valid["Score composite"] = (valid["Sim. intra"] - valid["Sim. inter"] + valid["MRR"] + valid["NDCG"]).round(3)
emb_ranking = valid.sort_values("Score composite", ascending=False)
print(emb_ranking[["Sim. intra", "Sim. inter", "MRR", "NDCG", "Score composite"]].to_string())
best_embedding_name = emb_ranking.index[0]
print(f"\\n>>> Meilleure méthode d'embedding : {best_embedding_name}")

emb_pct_metrics = ["Cosine sim. (moy)", "Sim. intra", "Sim. inter", "Precision@2", "Recall@2", "MAP", "NDCG"]
med_e = valid[emb_pct_metrics].median()
emb_pct = pd.DataFrame(index=valid.index)
for m in emb_pct_metrics:
    sign = -1.0 if m == "Sim. inter" else 1.0
    denom = med_e[m] if med_e[m] != 0 else 1e-9
    emb_pct[f"{m} %"] = ((sign * (valid[m] - med_e[m])) / abs(denom) * 100).round(1)
emb_results_df_pct = pd.concat([valid, emb_pct], axis=1)
emb_results_df_pct
"""))

cells.append(code("""# Justification écrite de la meilleure méthode d'embedding
w = emb_results_df_pct.loc[best_embedding_name]
med = emb_results_df_pct[["Sim. intra", "Sim. inter", "MRR", "NDCG"]].median()
second = emb_ranking.index[1]

embedding_justification = (
    f"« {best_embedding_name} » domine avec un score composite de {w['Score composite']:.3f} "
    f"(+{w['Score composite'] - emb_ranking['Score composite'].iloc[1]:.3f} devant « {second} »). "
    f"Ses pseudo-questions retrouvent leur chunk au rang moyen {1/w['MRR']:.1f} (MRR = {w['MRR']:.3f}, "
    f"{(w['MRR']/med['MRR']-1)*100:+.1f} % vs médiane) et son classement est le plus fiable "
    f"(nDCG = {w['NDCG']:.3f}, soit {(w['NDCG']/med['NDCG']-1)*100:+.1f} %). "
    f"Cela confirme l'hypothèse de départ : un espace vectoriel pré-entraîné sur de très grands corpus "
    f"généralise bien au-delà de notre thèse de 100 pages, là où les modèles entraînés from scratch "
    f"(Word2Vec, FastText, Doc2Vec) n'ont vu que ~{len(corpus_text.split()):,} mots."
)
print(embedding_justification)
"""))

cells.append(code("""# Visualisation : heatmaps des similarités chunk x chunk par méthode + barres MRR
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ok_methods = [m for m in emb_ranking.index if m in embed_chunk_embs]
emb_colors = ["#7209b7" if m == best_embedding_name else "#4c78a8" for m in emb_ranking.index]
emb_ranking["MRR"].plot(kind="barh", ax=axes[0], color=emb_colors, title="MRR par méthode d'embedding")
axes[0].invert_yaxis()

mats = {}
for m in ok_methods[:4]:
    mats[m] = cosine_similarity(embed_chunk_embs[m][:60])
vmax = max(float(m.max()) for m in mats.values())
im = axes[1].imshow(np.mean([mats[m] for m in mats], axis=0), cmap="viridis", vmin=0, vmax=max(0.3, vmax))
axes[1].set_title("Similarité chunk×chunk (moyenne des 4 meilleures, 60 chunks)")
plt.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout()
plt.savefig("docs/embedding_compare.png", dpi=130)
plt.show()
"""))

cells.append(code("""# Export JSON de la comparaison d'embedding
embedding_export = {
    "stage": "embedding",
    "metrics_columns": ["Cosine sim. (moy)", "Sim. intra", "Sim. inter", "Precision@2",
                        "Recall@2", "MAP", "MRR", "NDCG", "Score composite"],
    "higher_is_better": {"Cosine sim. (moy)": True, "Sim. intra": True, "Sim. inter": False,
                         "Precision@2": True, "Recall@2": True, "MAP": True, "MRR": True,
                         "NDCG": True, "Score composite": True},
    "methods": [
        {"name": idx, **{col: (float(row[col]) if row[col] is not None and not pd.isna(row[col]) else None)
                          for col in emb_results_df_pct.columns}}
        for idx, row in emb_results_df_pct.iterrows()
    ],
    "best": {"name": best_embedding_name, "justification": embedding_justification},
    "ranking": list(emb_ranking.index),
}
with open("results/embedding.json", "w", encoding="utf-8") as f:
    json.dump(embedding_export, f, ensure_ascii=False, indent=2, default=str)
print("results/embedding.json écrit.")
"""))

part3 = cells

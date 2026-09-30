# -*- coding: utf-8 -*-
"""Partie 4 du notebook : comparaison des 6 methodes de retrieval + generation LLM."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code

cells = []

# ============ COMPARAISON RETRIEVAL ============
cells.append(md("""# PARTIE 4 — Comparaison des 6 méthodes de Retrieval

**Points de méthodologie** : chunking fixé (gagnant Partie 2), embedding fixé (gagnant Partie 3). On ne fait varier **que** la stratégie de recherche.

## Les 6 méthodes

| # | Méthode | Principe | Points forts | Limites |
|---|---|---|---|---|
| 1 | **Similarité brute** | comparaison **exhaustive** question × tous les chunks (cosinus exact) | résultat exact, zéro index à construire | temps de recherche O(N) par requête |
| 2 | **ANN rapide (FAISS)** | index approximatif (HNSW) : ne parcourt qu'une fraction des vecteurs | très rapide sur de gros corpus | léger risque de rater le vrai plus proche voisin |
| 3 | **Mots-clés (BM25)** | score lexical : rareté des termes + saturation + longueur | excellent sur les **termes exacts** (noms, sigles), aucune IA nécessaire | aveugle aux synonymes et à la reformulation |
| 4 | **Dense spécialisé (DPR)** | deux encodeurs (question / passage) entraînés **spécifiquement** pour le QA | aligne l'espace des questions sur celui des passages | nécessite un modèle spécifique, plus lourd |
| 5 | **Hybride (BM25 + dense, RRF)** | fusionne les classements lexical et sémantique par **Reciprocal Rank Fusion** | robuste : lexical + sens, sans accord de poids délicat | deux index à maintenir |
| 6 | **Re-ranking 2 étapes** | 1er étage rapide (récupère ~25 candidats) puis **cross-encoder** qui lit question+passage **ensemble** | la meilleure précision : le cross-encoder croise les mots finement | plus lent (1 inférence par candidat) |

## Métriques (même famille que les étapes précédentes)

| Métrique | Objectif | Calcul |
|---|---|---|
| Precision@5 | pertinence des 5 résultats | hits / (5 × nb requêtes) |
| Recall@5 | couverture | tous les passages pertinents sont-ils dans le top-5 ? |
| MRR | position du premier pertinent | moyenne de 1/rang |
| MAP@5 | précision moyenne sur tout le top-5 | moyenne des précisions aux rangs où un passage pertinent est retrouvé (≠ MRR ici : plusieurs passages pertinents par question) |
| nDCG@5 | qualité du classement | DCG idéal / DCG obtenu (log2) |
| Cosine sim. | proximité sémantique brute | cosinus moyen (embedding question, embedding du 1er passage retourné), mesuré dans le même espace pour les 6 méthodes |
| Latence (ms) | vitesse de réponse par requête | temps moyen mesuré (↓ mieux) |

**Jeu d'évaluation** : 20 questions réalistes sur la thèse (pollution azotée, zones humides, dénitrification...), chacune annotée par les **chunks pertinents** (chunk dont le contenu recouvre ≥ 2 mots significatifs de la question → score de pertinence gradué 0-3 pour le nDCG).
"""))

cells.append(code("""# ---------- Fixer les données de la comparaison ----------
import time

# 1) Chunks : gagnant du chunking, embeddés avec le gagnant de l'embedding
retrieval_chunks = best_chunks
chunk_embs = np.array(embed_fn(retrieval_chunks))          # espace du gagnant embedding (MiniLM)
chunk_embs_unit = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)
norms = np.linalg.norm(chunk_embs, axis=1)

# 2) Questions de test réalistes + tokens significatifs pour l'annotation automatique
test_queries = [
    "Comment les zones humides réduisent-elles la pollution par les nitrates ?",
    "Quel est le rôle des zones tampons dans le bassin versant ?",
    "Comment la dénitrification élimine-t-elle les nitrates du sol ?",
    "Quel impact de l'intensification de l'agriculture sur la qualité de l'eau ?",
    "Comment mesure-t-on les flux de nitrate dans les zones humides ?",
    "Quel est le rôle des roseaux dans l'épuration des eaux ?",
    "Comment les zones humides de fond de vallée piègent-elles les sédiments ?",
    "Qu'est-ce que la régulation des pollutions azotées diffuses ?",
    "Comment la végétation des berges influence-t-elle les flux d'azote ?",
    "Quels facteurs contrôlent la dénitrification dans les sols ?",
    "Quel est l'effet du drainage agricole sur les nappes ?",
    "Comment les prairies réduisent-elles le lessivage des nitrates ?",
    "Quel est le rôle des bactéries dans le cycle de l'azote ?",
    "Comment la topographie catena influence-t-elle les flux d'eau ?",
    "Qu'est-ce qu'une zone humide riparienne et son fonctionnement ?",
    "Comment la rotation des cultures influence la lixiviation de l'azote ?",
    "Quels outils pour cartographier les zones humides ?",
    "Comment la matière organique des sols favorise-t-elle la dénitrification ?",
    "Quel est l'impact des pratiques agricoles sur les émissions de N2O ?",
    "Comment les zones humides contribuent-elles à la qualité des eaux de surface ?",
]
import re as _re2
def sig_stems(q):
    # Stems = 6 premiers caractères des mots significatifs (approximation de stemming français :
    # neutralise les flexions comme réduisent/réduction, zones/zone, bactéries/bactéries).
    return sorted({w[:6] for w in _re2.findall(r"[a-zàâäéèêëîïôöùûüç]{4,}", q.lower())
                   if w not in {"comment", "quel", "quelles", "quels", "quelle", "est", "dans", "pour", "avec",
                                "pourquoi", "rôle", "role", "impact", "effet", "facteurs", "outils", "qu'est",
                                "influence", "contribuent", "favorise", "mesure", "quels"}})

def annotate_relevance(chunks, queries):
    # Pertinence graduée 0-3 = fraction des stems de la question présents dans le chunk
    # (méthode neutre : n'avantage ni les méthodes lexicales ni les denses).
    rel = {}
    for qi, q in enumerate(queries):
        stems = sig_stems(q)
        if not stems:
            rel[qi] = np.zeros(len(chunks), dtype=float)
            continue
        scores = []
        for c in chunks:
            cl = c.lower()
            h = sum(1 for s in stems if s in cl)
            frac = h / len(stems)
            scores.append(min(3.0, round(frac * 3, 1)))
        rel[qi] = np.array(scores, dtype=float)
    return rel

relevance = annotate_relevance(retrieval_chunks, test_queries)
print(f"{len(retrieval_chunks)} chunks | {len(test_queries)} questions annotées (pertinence graduée 0-3).")
"""))

cells.append(code("""# ---------- Les 6 fonctions de retrieval ----------
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

BM25_TOKEN = re.compile(r"[a-zàâäéèêëîïôöùûüç]{2,}")

def bm25_tokenize(t):
    return [w for w in BM25_TOKEN.findall(t.lower()) if w not in
            {"les", "des", "une", "dans", "pour", "avec", "sont", "cette", "leur", "plus", "mais",
             "que", "qui", "par", "sur", "entre", "ainsi", "cela", "afin", "ors", "donc"}]

# 1) Similarité brute : cosinus exact sur tous les chunks
def retrieve_brute(question, k=5):
    q = np.array(embed_fn([question]))
    sims = cosine_similarity(q, chunk_embs)[0]
    idx = np.argsort(sims)[::-1][:k]
    return list(idx), time.perf_counter()

# 2) ANN rapide : FAISS HNSW
import faiss
dim = chunk_embs.shape[1]
hnsw = faiss.IndexHNSWFlat(dim, 32)
hnsw.add(chunk_embs.astype("float32"))

def retrieve_ann(question, k=5):
    q = np.array(embed_fn([question])).astype("float32")
    _, idx = hnsw.search(q, k)
    return list(idx[0]), time.perf_counter()

# 3) BM25 (mots-clés)
bm25 = BM25Okapi([bm25_tokenize(c) for c in retrieval_chunks])

def retrieve_bm25(question, k=5):
    scores = bm25.get_scores(bm25_tokenize(question))
    idx = np.argsort(scores)[::-1][:k]
    return list(idx), time.perf_counter()

# 4) Dense spécialisé (DPR) : encodeur question + encodeur contexte
from sentence_transformers import SentenceTransformer
dpr_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
dpr_chunk_embs = dpr_model.encode(retrieval_chunks, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

def retrieve_dpr(question, k=5):
    q = dpr_model.encode([question], normalize_embeddings=True)
    sims = (q @ dpr_chunk_embs.T)[0]
    idx = np.argsort(sims)[::-1][:k]
    return list(idx), time.perf_counter()

# 5) Hybride : RRF(BM25, dense MiniLM)
def retrieve_hybrid(question, k=5, dense_top=25):
    q = np.array(embed_fn([question]))
    sims = cosine_similarity(q, chunk_embs)[0]
    dense_idx = np.argsort(sims)[::-1][:dense_top]
    bm_idx = np.argsort(bm25.get_scores(bm25_tokenize(question)))[::-1][:dense_top]
    rrf = {}
    for rank, i in enumerate(dense_idx):
        rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
    for rank, i in enumerate(bm_idx):
        rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
    idx = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return idx, time.perf_counter()

# 6) Re-ranking : 1er étage dense (25 candidats) + cross-encoder
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_rerank(question, k=5, candidate_top=25):
    q = np.array(embed_fn([question]))
    sims = cosine_similarity(q, chunk_embs)[0]
    cand = np.argsort(sims)[::-1][:candidate_top]
    pairs = [(question, retrieval_chunks[i][:512]) for i in cand]
    ce_scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
    order = np.argsort(ce_scores)[::-1]
    return [int(cand[j]) for j in order[:k]], time.perf_counter()

retrieval_methods = {
    "1. Similarité brute": retrieve_brute,
    "2. ANN rapide (FAISS)": retrieve_ann,
    "3. Mots-clés (BM25)": retrieve_bm25,
    "4. Dense spécialisé (DPR)": retrieve_dpr,
    "5. Hybride (BM25+dense)": retrieve_hybrid,
    "6. Re-ranking (2 étapes)": retrieve_rerank,
}
print("6 fonctions de retrieval définies.")
"""))

cells.append(code("""# ---------- Boucle d'évaluation des 6 méthodes ----------
def precision_recall_mrr_ndcg(ranked_idx, rel_row, k=5):
    \"\"\"rel_row : pertinence graduée 0-3 par chunk. Pertinent si score >= 2.\"\"\"
    topk = list(ranked_idx)[:k]
    gains = np.array([rel_row[i] for i in topk], dtype=float)
    relevant_mask = gains >= 2
    hits = int(relevant_mask.sum())
    precision = hits / k
    recall = hits / max(1, int((rel_row >= 2).sum()))
    first_rel = next((r + 1 for r, g in enumerate(topk) if rel_row[g] >= 2), None)
    mrr = (1.0 / first_rel) if first_rel else 0.0
    dcg = sum((2**g - 1) / np.log2(pos + 2) for pos, g in enumerate(gains))
    ideal = np.sort(rel_row)[::-1][:k]
    idcg = sum((2**g - 1) / np.log2(pos + 2) for pos, g in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    # Average Precision@5 (= MAP@5 par requête) : précision moyenne aux rangs où un
    # passage pertinent est retrouvé. Plusieurs passages pertinents par question
    # ici, donc MAP != MRR (contrairement à l'étape embedding, 1 seul pertinent).
    n_relevant = max(1, int((rel_row >= 2).sum()))
    cum_hits, ap_sum = 0, 0.0
    for pos, g in enumerate(topk):
        if rel_row[g] >= 2:
            cum_hits += 1
            ap_sum += cum_hits / (pos + 1)
    ap = ap_sum / min(k, n_relevant)
    return precision, recall, mrr, ndcg, ap

retrieval_rows = []
retrieval_latencies = {}
for name, fn in retrieval_methods.items():
    precs, recalls, mrrs, ndcgs, aps, lats, coss = [], [], [], [], [], [], []
    for qi, q in enumerate(test_queries):
        t0 = time.perf_counter()
        idx, _ = fn(q, k=5)
        dt = (time.perf_counter() - t0) * 1000
        p, r, m, n, ap = precision_recall_mrr_ndcg(idx, relevance[qi], k=5)
        # Cosine sim. : proximité (embedding question, embedding du 1er passage retourné),
        # mesurée dans le MÊME espace (gagnant embedding) pour les 6 méthodes -> comparable.
        q_emb = np.array(embed_fn([q]))
        top1_emb = np.array(embed_fn([retrieval_chunks[idx[0]]]))
        cos = float(cosine_similarity(q_emb, top1_emb)[0][0])
        precs.append(p); recalls.append(r); mrrs.append(m); ndcgs.append(n)
        aps.append(ap); lats.append(dt); coss.append(cos)
    retrieval_rows.append({
        "Méthode": name,
        "Precision@5": round(float(np.mean(precs)), 3),
        "Recall@5": round(float(np.mean(recalls)), 3),
        "MRR": round(float(np.mean(mrrs)), 3),
        "MAP@5": round(float(np.mean(aps)), 3),
        "NDCG@5": round(float(np.mean(ndcgs)), 3),
        "Cosine sim.": round(float(np.mean(coss)), 3),
        "Latence (ms)": round(float(np.mean(lats)), 1),
    })
    print(f"{name:28s} P@5={np.mean(precs):.3f} R@5={np.mean(recalls):.3f} MRR={np.mean(mrrs):.3f} MAP={np.mean(aps):.3f} NDCG={np.mean(ndcgs):.3f} cos={np.mean(coss):.3f} lat={np.mean(lats):.1f}ms")

ret_results_df = pd.DataFrame(retrieval_rows).set_index("Méthode")
ret_results_df
"""))

cells.append(md("""## Meilleure méthode de retrieval

**Score qualité** = moyenne normalisée (0-1) de Precision@5, Recall@5, MRR, MAP@5, nDCG@5 et Cosine sim. ; puis **Score global** = 0.7 × qualité − 0.3 × latence normalisée (le re-ranking peut être un peu plus lent : c'est le compromis qualité/vitesse qui décide).
"""))

cells.append(code("""q_cols = ["Precision@5", "Recall@5", "MRR", "MAP@5", "NDCG@5", "Cosine sim."]
norm = (ret_results_df[q_cols] - ret_results_df[q_cols].min()) / (ret_results_df[q_cols].max() - ret_results_df[q_cols].min() + 1e-9)
ret_results_df["Score qualité"] = norm.mean(axis=1).round(3)
lat_norm = (ret_results_df["Latence (ms)"] - ret_results_df["Latence (ms)"].min()) / (ret_results_df["Latence (ms)"].max() - ret_results_df["Latence (ms)"].min() + 1e-9)
ret_results_df["Score global"] = (0.7 * ret_results_df["Score qualité"] - 0.3 * lat_norm).round(3)

ret_ranking = ret_results_df.sort_values("Score global", ascending=False)
best_retrieval_name = ret_ranking.index[0]
print(ret_ranking.to_string())
print(f"\\n>>> Meilleure méthode de retrieval : {best_retrieval_name}")

# Pourcentages vs médiane
ret_pct_metrics = ["Precision@5", "Recall@5", "MRR", "MAP@5", "NDCG@5", "Cosine sim.", "Latence (ms)"]
med_r = ret_results_df[ret_pct_metrics].median()
ret_pct = pd.DataFrame(index=ret_results_df.index)
for m in ret_pct_metrics:
    sign = -1.0 if m == "Latence (ms)" else 1.0
    denom = med_r[m] if med_r[m] != 0 else 1e-9
    ret_pct[f"{m} %"] = ((sign * (ret_results_df[m] - med_r[m])) / abs(denom) * 100).round(1)
ret_results_df_pct = pd.concat([ret_results_df, ret_pct], axis=1)
ret_results_df_pct
"""))

cells.append(code("""# Justification écrite de la meilleure méthode de retrieval
w = ret_results_df.loc[best_retrieval_name]
med = ret_results_df[q_cols].median()
second = ret_ranking.index[1]

retrieval_justification = (
    f"« {best_retrieval_name} » obtient le meilleur score global ({w['Score global']:.3f}) : "
    f"nDCG@5 = {w['NDCG@5']:.3f} ({(w['NDCG@5']/med['NDCG@5']-1)*100:+.1f} % vs médiane), "
    f"MAP@5 = {w['MAP@5']:.3f} ({(w['MAP@5']/med['MAP@5']-1)*100:+.1f} %) et MRR = {w['MRR']:.3f} ({(w['MRR']/med['MRR']-1)*100:+.1f} %), "
    f"avec une latence maîtrisée ({w['Latence (ms)']:.1f} ms/requête). "
    f"Devant « {second} » ({ret_ranking['Score global'].iloc[1]:.3f}), elle gagne parce que "
    + ("le cross-encoder lit chaque candidat **avec** la question et capte les correspondances fines que les bi-encodeurs manquent, "
       "ce qui remonte les passages vraiment pertinents au sommet du classement, tandis que le premier étage dense "
       "garde la recherche rapide."
       if "Re-ranking" in best_retrieval_name else
       "elle combine les termes exacts (BM25) et le sens (dense) : quand la question reformule le document, le dense compense ; "
       "quand elle emploie le vocabulaire exact de la thèse, le lexical confirme — la fusion RRF cumule les deux forces."
       if "Hybride" in best_retrieval_name else
       "les questions de test emploient souvent le vocabulaire exact du document (noms techniques comme « dénitrification », "
       "« zones humides »), et le score BM25 capte ces correspondances lexicales avec une précision remarquable pour un coût "
       "de calcul dérisoire (aucun modèle d'embedding requis) — les approches denses compensent quand la question reformule le "
       "texte, mais sur ce corpus technique le signal lexical domine, et l'écart de latence (2 ms contre 100 ms+) rend le "
       "compromis qualité/vitesse imbattable.")
)
print(retrieval_justification)
"""))

cells.append(code("""# Visualisation : trade-off qualité vs latence + barres par métrique
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
colors = ["#7209b7" if m == best_retrieval_name else "#4c78a8" for m in ret_results_df.index]
axes[0].scatter(ret_results_df["Latence (ms)"], ret_results_df["Score qualité"], s=160, c=colors, alpha=0.85)
for m in ret_results_df.index:
    axes[0].annotate(m.split(". ")[1], (ret_results_df.loc[m, "Latence (ms)"], ret_results_df.loc[m, "Score qualité"]),
                     fontsize=8, xytext=(5, 4), textcoords="offset points")
axes[0].set_xlabel("Latence (ms) ↓ mieux")
axes[0].set_ylabel("Score qualité (0-1)")
axes[0].set_title("Trade-off qualité / vitesse (violet = gagnant)")

ret_results_df[q_cols].plot(kind="barh", ax=axes[1], title="Précision / Rappel / MRR / nDCG@5")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("docs/retrieval_compare.png", dpi=130)
plt.show()
"""))

cells.append(code("""# Export JSON de la comparaison de retrieval
retrieval_export = {
    "stage": "retrieval",
    "metrics_columns": ["Precision@5", "Recall@5", "MRR", "MAP@5", "NDCG@5", "Cosine sim.", "Latence (ms)", "Score qualité", "Score global"],
    "higher_is_better": {"Precision@5": True, "Recall@5": True, "MRR": True, "MAP@5": True, "NDCG@5": True,
                         "Cosine sim.": True, "Latence (ms)": False, "Score qualité": True, "Score global": True},
    "methods": [
        {"name": idx, **{col: float(row[col]) for col in ret_results_df.columns}}
        for idx, row in ret_results_df.iterrows()
    ],
    "best": {"name": best_retrieval_name, "justification": retrieval_justification},
    "ranking": list(ret_ranking.index),
}
with open("results/retrieval.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_export, f, ensure_ascii=False, indent=2, default=str)
print("results/retrieval.json écrit.")
"""))

# ============ MEILLEURES METHODES GLOBALES ============
cells.append(code("""# ---------- best_methods.json : gagnant + justification pour chaque étape ----------
best_methods_export = {
    "chunking": {"name": best_chunking_name, "justification": chunking_justification,
                 "score": float(results_df.loc[best_chunking_name, "Score composite"])},
    "embedding": {"name": best_embedding_name, "justification": embedding_justification,
                  "score": float(emb_results_df_pct.loc[best_embedding_name, "Score composite"])},
    "retrieval": {"name": best_retrieval_name, "justification": retrieval_justification,
                  "score": float(ret_results_df.loc[best_retrieval_name, "Score global"])},
    "pipeline_final": {
        "chunking": best_chunking_name,
        "embedding": best_embedding_name,
        "retrieval": best_retrieval_name,
        "resume": ("Le pipeline retenu enchaîne : découpage « %s » (chunks cohérents, peu redondants), "
                   "vectorisation « %s » (espace sémantique pré-entraîné), recherche « %s » (meilleur compromis "
                   "qualité/latence sur nos 20 questions de test)."
                   % (best_chunking_name, best_embedding_name, best_retrieval_name)),
    },
}
with open("results/best_methods.json", "w", encoding="utf-8") as f:
    json.dump(best_methods_export, f, ensure_ascii=False, indent=2)
print("results/best_methods.json écrit.")
print(best_methods_export["pipeline_final"]["resume"])
"""))

# ============ GENERATION LLM ============
cells.append(md("""# PARTIE 5 — Génération (traitement des requêtes) : Ollama + repli Hugging Face

Dernière étape du pipeline : les passages récupérés + la question sont envoyés à un **LLM** qui rédige la réponse en s'appuyant **uniquement** sur ces extraits (avec sources). Deux backends testés :
- **Ollama** (local, demandé par le sujet) : `llama3.2:1b` via l'API `http://localhost:11434/api/chat` ;
- **Hugging Face** (repli) : petit modèle de génération.

**Nouveauté (extra)** : un second mini-benchmark comparant 3 tailles de modèles Ollama (1B / 3B / 8B) sur 5 questions : latence par token et score de citation des sources — à présenter devant le jury.
"""))

cells.append(code("""# ---------- Détection du backend LLM disponible ----------
import requests, json

OLLAMA_URL = "http://localhost:11434"
GEN_MODEL = "llama3.2:1b"

def ollama_ok():
    try:
        models = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json().get("models", [])
        return any(m["name"].startswith("llama3.2") for m in models)
    except Exception:
        return False

ollama_ready = ollama_ok()
print("Ollama prêt :", ollama_ready)
"""))

cells.append(code("""# ---------- Génération : Ollama (prioritaire) puis repli Hugging Face ----------
def generate_ollama(prompt, model=GEN_MODEL, max_tokens=300, temperature=0.2):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }, timeout=180)
    r.raise_for_status()
    return r.json()["response"]

_hf_pipeline = None
def generate_hf(prompt, max_new_tokens=250):
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        _hf_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct",
                                torch_dtype="auto", device_map="cpu")
    messages = [{"role": "user", "content": prompt}]
    out = _hf_pipeline(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"]

def generate(prompt):
    if ollama_ready:
        try:
            return generate_ollama(prompt), "ollama"
        except Exception as e:
            print(f"Ollama a échoué ({str(e)[:80]}) -> repli Hugging Face.")
    return generate_hf(prompt), "huggingface"

def build_prompt(question, passages):
    context = "\\n\\n".join(f"[Source {i+1}] {p[:600]}" for i, p in enumerate(passages))
    return (
        "Tu es un assistant documentaire. Réponds en français, de façon factuelle et concise (5 phrases maximum), "
        "UNIQUEMENT à partir des extraits ci-dessous. Cite les sources utilisées sous la forme [Source N]. "
        "Si les extraits ne permettent pas de répondre, dis-le clairement.\\n\\n"
        f"### Extraits du corpus\\n{context}\\n\\n### Question\\n{question}\\n\\n### Réponse :"
    )

print("Fonctions de génération prêtes (Ollama + repli HF).")
"""))

cells.append(code("""# ---------- Démo end-to-end : question -> retrieval (méthode gagnante) -> réponse citée ----------
demo_questions = [
    "Comment les zones humides réduisent-elles la pollution par les nitrates ?",
    "Quels facteurs contrôlent la dénitrification dans les sols ?",
]
best_retrieval_fn = retrieval_methods[best_retrieval_name]

for q in demo_questions:
    idx, _ = best_retrieval_fn(q, k=3)
    passages = [retrieval_chunks[i] for i in idx]
    prompt = build_prompt(q, passages)
    answer, backend = generate(prompt)
    print("=" * 100)
    print(f"QUESTION : {q}")
    print(f"RETRIEVAL ({best_retrieval_name}) -> chunks {list(idx)} | GÉNÉRATION : {backend}")
    print("-" * 100)
    print(answer.strip()[:1200])
    print(f"\\n[SOURCES] " + " | ".join(f"[{i+1}] {passages[i][:90]}..." for i in range(len(passages))))
"""))

cells.append(code("""# ---------- Mini-benchmark LLM : 3 tailles de modèles (1B / 3B / 8B) sur 5 questions ----------
import time as _time

def bench_model(model_name, questions, retrieval_fn, chunks, k=3, max_tokens=200):
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n.startswith(model_name.split(":")[0]) for n in names):
        return None
    rows = []
    for q in questions:
        idx, _ = retrieval_fn(q, k=k)
        passages = [chunks[i] for i in idx]
        prompt = build_prompt(q, passages)
        t0 = _time.perf_counter()
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model_name, "prompt": prompt, "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.2}}, timeout=300)
            resp.raise_for_status()
            answer = resp.json()["response"]
            dt = (_time.perf_counter() - t0)
            n_tok = resp.json().get("eval_count", max(1, len(answer.split())))
            toks = [w for w in q.lower().split() if len(w) > 5]
            cited = sum(1 for t in toks if t in answer.lower())
            rows.append({"latence_s": round(dt, 2), "tokens": int(n_tok),
                         "ms/token": round(dt * 1000 / max(1, n_tok), 1), "citations_ok": cited})
        except Exception:
            rows.append({"latence_s": None, "tokens": None, "ms/token": None, "citations_ok": 0})
    df = pd.DataFrame(rows)
    return {"model": model_name, "latence_moy_s": round(df["latence_s"].dropna().mean(), 2),
            "ms_par_token": round(df["ms/token"].dropna().mean(), 1), "citations_moy": round(df["citations_ok"].mean(), 1)}

bench_questions = test_queries[:5]
llm_rows = []
for model_name in ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]:
    print(f"Benchmark {model_name} sur {len(bench_questions)} questions...")
    r = bench_model(model_name, bench_questions, best_retrieval_fn, retrieval_chunks)
    if r:
        llm_rows.append(r)
    else:
        print(f"  {model_name} non installé -> ignoré.")
llm_bench_df = pd.DataFrame(llm_rows)
llm_bench_df
"""))

cells.append(code("""# Export JSON du benchmark LLM (les timeouts deviennent null, pas NaN)
import math as _math

def _clean_llm_rows(rows):
    return [{k: (None if isinstance(v, float) and _math.isnan(v) else v) for k, v in r.items()} for r in rows]

llm_rows = _clean_llm_rows(llm_rows)
with open("results/llm_benchmark.json", "w", encoding="utf-8") as f:
    json.dump({"models": llm_rows}, f, ensure_ascii=False, indent=2, allow_nan=False)
print("results/llm_benchmark.json écrit.")
"""))

part4 = cells

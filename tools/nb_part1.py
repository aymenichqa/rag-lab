# -*- coding: utf-8 -*-
"""Partie 1 du notebook : intro, concepts RAG, cleaning du PDF -> corpus."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code

cells = []

# ============ EN-TETE ============
cells.append(md("""# Projet RAG — Pipeline complet avec comparaison des méthodes

**PFA — Stage d'été — 4ème Année IIR — Encadrante : Dr. EL MKHALET MOUNA**

Ce notebook est la **source de vérité** du projet : il implémente et exécute de bout en bout toutes les étapes du pipeline RAG sur un corpus réel (thèse académique de 191 pages), puis compare les méthodes de chaque étape :

1. **Cleaning du Data** (PDF brut → corpus ~100 pages, conforme au guide de l'encadrante)
2. **Concepts RAG** : principe, utilité, cas d'usage, avantages, étapes clés
3. **Chunking** : comparaison de **7 méthodes** (8 métriques)
4. **Embedding** : comparaison de **7 méthodes** (8 métriques)
5. **Retrieval** : comparaison de **6 méthodes** (précision@5, rappel@5, MRR, MAP@5, nDCG@5, cosine sim., latence)
6. **Génération (LLM)** : démo Ollama + repli Hugging Face
7. **Architecture du projet** : schéma généré + export
8. **Exports JSON** : consommés par l'application web (FastAPI + React)

**Prérequis d'exécution** :
- Environnement : `pip install -r requirements.txt` (Python 3.13 recommandé)
- Corpus brut : `thèse_pdf.pdf` dans le même dossier que ce notebook
- Optionnel (méthode 7 d'embedding + démo génération) : Ollama lancé (`ollama serve`) avec les modèles `nomic-embed-text` et `llama3.2:1b` (`ollama pull <modele>`)
- Première exécution avec internet : téléchargement des modèles pré-entraînés (GloVe ~130 Mo, BERT ~440 Mo, MiniLM ~90 Mo, cross-encoder ~440 Mo)

---
"""))

# ============ CONCEPTS RAG ============
cells.append(md("""# PARTIE 0 — Concepts fondamentaux du RAG (Retrieval-Augmented Generation)

> *Cellule académique demandée par le sujet : comprendre avant d'implémenter.*

## Principe de fonctionnement

Un LLM (Large Language Model) génère des réponses à partir de ses connaissances **figées à sa date d'entraînement**. Il ignore tout de vos documents internes (rapports, thèses, bases de connaissances) et peut « halluciner » (inventer des faits plausibles). Le **RAG** corrige ces deux limites en deux temps :

```
                    ┌────────────────────────  HORS-LIGNE (indexation)  ────────────────────────┐
 Document(s) ──► CLEANING ──► CHUNKING (découpage) ──► EMBEDDING (vectorisation) ──► INDEX vectoriel
                    └────────────────────────────────────────────────────────────────────────┘
                    ┌────────────────────────── EN-LIGNE (requête)  ──────────────────────────┐
 Question ──► EMBEDDING de la question ──► RETRIEVAL (recherche des top-k passages) ──►
            ──► (optionnel : RE-RANKING) ──► PROMPT = question + passages ──► LLM ──► RÉPONSE + SOURCES
                    └────────────────────────────────────────────────────────────────────────┘
```

- **Retrieval** : avant de répondre, le système **cherche** les passages les plus pertinents du corpus (par similarité sémantique, mots-clés, ou les deux).
- **Augmented** : les passages récupérés sont **injectés dans le prompt** du LLM.
- **Generation** : le LLM rédige sa réponse **uniquement à partir de ces passages** (approche « à livre ouvert »), ce qui réduit les hallucinations et permet de **citer les sources**.

## Utilité et cas d'usage

| Cas d'usage | Exemple concret |
|---|---|
| Assistance documentaire | chatbot interne branché sur la documentation d'entreprise |
| Veille scientifique | interroger un corpus de thèses/articles (notre cas) |
| Support client | FAQ augmentée des tickets historiques |
| Conformité / juridique | retrouver les clauses exactes dans des contrats |
| Éducation | tutor interactif sur les polycopiés d'un cours |

## Avantages du RAG

1. **Fraîcheur** : le corpus change, pas besoin de ré-entraîner le modèle — il suffit de ré-indexer.
2. **Traçabilité** : chaque réponse cite les passages sources (vérifiable par l'humain).
3. **Réduction des hallucinations** : le LLM est contraint de s'appuyer sur des extraits réels.
4. **Coût** : bien moins cher qu'un fine-tuning, et compatible avec des LLM locaux (Ollama) → données privées.
5. **Contrôle d'accès** : on peut filtrer les documents indexés par utilisateur/rôle.

## Les 4 étapes clés (celles comparées dans ce projet)

1. **Découpage (chunking)** : découper le corpus en morceaux (« chunks ») assez courts pour porter une idée unique. Trop gros → l'embedding dilue le sens ; trop petits → le contexte est perdu. **7 méthodes comparées en Partie 2.**
2. **Vectorisation (embedding)** : transformer chaque chunk en vecteur numérique qui capture son **sens** (deux textes proches sémantiquement ont des vecteurs proches). **7 méthodes comparées en Partie 3.**
3. **Recherche d'information (retrieval)** : retrouver les top-k chunks les plus pertinents pour une question (exact vs approximatif, lexical vs sémantique, 1 ou 2 étages). **6 méthodes comparées en Partie 4.**
4. **Traitement des requêtes (query)** : construire le prompt final (question + passages) et générer la réponse avec citations — **démontré en Partie 5 avec Ollama / Hugging Face**.

---

"""))

cells.append(md("""## Principales bibliothèques RAG

*Demandée par le jury (panneau 1) : connaître les principales bibliothèques du domaine RAG et leur rôle dans ce projet.*

| Bibliothèque | Rôle dans un pipeline RAG | Où dans ce projet |
|---|---|---|
| **pypdf** | extraction du texte des PDF, page par page | Partie 1 (cleaning) |
| **LangChain** — `langchain-text-splitters`, `langchain-community`, `langchain-huggingface` | découpage du texte (splitters), connecteurs d'embeddings, vectorstores (FAISS) | TP1 + Partie 2 (chunking) |
| **sentence-transformers** | modèles d'embedding pré-entraînés (MiniLM, SBERT, multi-qa) + cross-encoder (re-ranking) | Parties 2, 3 et 4 + moteur live de l'application |
| **gensim** | Word2Vec, FastText et Doc2Vec entraînés from scratch (baselines à comparer) | Partie 3 (embedding) |
| **faiss-cpu** | index vectoriel ANN (HNSW) : recherche approximative ultra-rapide | TP1 + Partie 4 (retrieval) |
| **rank_bm25** | score BM25 : recherche par mots-clés (lexicale) | Partie 4 (retrieval) |
| **scikit-learn** | similarité cosinus, métriques | toutes les comparaisons |
| **Ollama** | LLM local : génération des réponses sourcées + embeddings (nomic-embed-text) | Partie 5 + application |
| **FastAPI + Uvicorn** | backend : API REST (résultats, meilleures méthodes, requête live) | application web (TP3) |
| **React + Vite + Recharts** | frontend : interface interactive (graphiques, podium, console de question) | application web (TP3) |

---

"""))

# ============ PARTIE CLEANING ============
cells.append(md("""# PARTIE 1 — Cleaning du Data : PDF brut → corpus ~100 pages

**Consignes de l'encadrante (Catégorie 2 — sujet entreprise + volet académique)** :
- I. Cleaning : supprimer la partie remerciements, le sommaire, la liste des figures et la liste des tableaux.
- II. Traitement : après le cleaning, réduire le dataset à ~100 pages.

**Méthode** (automatisée, reproductible) :
1. Extraction texte page par page (`pypdf`).
2. Détection des pages de front-matter (remerciements, sommaire, listes) par motifs d'en-têtes + heuristique de « pages de liste » (lignes se terminant par des numéros de page).
2bis. Détection de la couverture d'archive (page « HAL Id: … » déposée sur archives-ouvertes.fr) parmi les 3 premières pages — boilerplate administratif en anglais, sans valeur pour le RAG.
3. Détection du end-matter (bibliographie, annexes) — peu utile pour un RAG — à partir de la moitié du document.
4. Troncature au budget de 100 pages (on garde le début : introduction générale et corps du rapport).
5. Statistiques avant/après + visualisation.
"""))

cells.append(code("""import re
import numpy as np
import pandas as pd
from pypdf import PdfReader

# ---------------- Détection des sections non pertinentes ----------------
SECTION_HEADERS = [
    r"REMERCIEMENTS?", r"SOMMAIRE", r"TABLE\\s+DES\\s+MATI[EÈ]RES",
    r"LISTE\\s+DES\\s+FIGURES", r"LISTE\\s+DES\\s+TABLEAUX", r"TABLE\\s+DES\\s+FIGURES",
    r"TABLE\\s+DES\\s+TABLEAUX", r"TABLE\\s+DES\\s+ILLUSTRATIONS",
    r"LISTE\\s+DES\\s+ILLUSTRATIONS", r"LISTE\\s+DES\\s+ABR[EÉ]VIATIONS",
    r"R[EÉ]SUM[EÉ]\\s*$", r"ABSTRACT\\s*$",
]
HEADER_RE = re.compile("|".join(SECTION_HEADERS), re.IGNORECASE)
# ligne de sommaire : se termine par des points de suite + numéro de page
TOC_LINE_RE = re.compile(r"(\\.{2,}\\s*\\d{1,4}\\s*$)|(\\s\\d{1,4}\\s*$)")
END_MATTER_RE = re.compile(r"^\\s*(BIBLIOGRAPHIE|R[EÉ]F[EÉ]RENCES|ANNEXES?|TABLE\\s+DES\\s+MATI[EÈ]RES\\s+D[EÉ]TAILL[EÉ]E)\\s*$", re.IGNORECASE)

HAL_COVER_RE = re.compile(r"HAL\\s+Id\\s*:", re.IGNORECASE)   # couverture d'archive (dépôt HAL) : boilerplate hors thèse

def extract_pages(path):
    reader = PdfReader(path)
    return [p.extract_text() or "" for p in reader.pages]

def looks_like_list_page(page_text, min_ratio=0.35):
    lines = [l for l in page_text.split("\\n") if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if TOC_LINE_RE.search(l.strip()))
    return (hits / len(lines)) >= min_ratio

def find_cover_pages(pages, max_first=3):
    # Couverture d'archive (ex. page HAL déposée sur archives-ouvertes.fr) :
    # boilerplate administratif en anglais, sans valeur pour le RAG.
    cover = set()
    for i, text in enumerate(pages[:max_first]):
        if HAL_COVER_RE.search(text[:400]):
            cover.add(i)
    return cover

def find_removable_pages(pages):
    removable, in_section = set(), False
    for i, text in enumerate(pages):
        head = "\\n".join(text.split("\\n")[:4])
        if HEADER_RE.search(head):
            in_section = True
            removable.add(i)
            continue
        if in_section:
            if looks_like_list_page(text):
                removable.add(i)
            else:
                in_section = False
    return removable

def find_end_matter_start(pages):
    for i, text in enumerate(pages):
        if i < int(len(pages) * 0.5):   # le end-matter ne commence jamais avant la moitié
            continue
        first_lines = [l.strip() for l in text.split("\\n")[:5] if l.strip()][:2]
        if any(END_MATTER_RE.match(l) for l in first_lines):
            return i
    return None

def clean_and_reduce(path, page_budget=100):
    pages = extract_pages(path)
    total = len(pages)
    removable = find_removable_pages(pages)
    cover = find_cover_pages(pages)
    removable |= cover
    end_start = find_end_matter_start(pages)
    if end_start is not None:
        removable.update(range(end_start, total))
    kept_idx = [i for i in range(total) if i not in removable]
    after_cleaning = len(kept_idx)
    truncated = False
    if len(kept_idx) > page_budget:
        kept_idx = kept_idx[:page_budget]
        truncated = True
    cleaned_text = "\\n\\n".join(pages[i] for i in kept_idx)
    return {
        "path": path, "total_pages": total,
        "cover_pages_removed": sorted(i + 1 for i in cover),
        "front_matter_removed": sorted(i + 1 for i in removable if end_start is None or i < end_start),
        "end_matter_start_page": (end_start + 1) if end_start is not None else None,
        "pages_after_cleaning": after_cleaning,
        "final_page_count": len(kept_idx),
        "truncated_to_budget": truncated,
        "text": cleaned_text,
    }

print("Fonctions de cleaning prêtes.")
"""))

cells.append(code("""PDF_PATH = "thèse_pdf.pdf"
PAGE_BUDGET = 100

info = clean_and_reduce(PDF_PATH, page_budget=PAGE_BUDGET)

print(f"Fichier                     : {info['path']}")
print(f"Pages totales (PDF brut)    : {info['total_pages']}")
print(f"Couverture archive retirée  : {info['cover_pages_removed']}")
print(f"Pages front-matter retirées : {info['front_matter_removed']}")
print(f"Bibliographie/annexes dès   : p.{info['end_matter_start_page']}")
print(f"Pages après cleaning        : {info['pages_after_cleaning']}")
print(f"Pages finales (budget {PAGE_BUDGET})   : {info['final_page_count']} (tronqué : {info['truncated_to_budget']})")
"""))

cells.append(code("""import os

corpus_text = info["text"]
os.makedirs("data", exist_ok=True)
with open("data/CORPUS_RAG_reduit_100pages.txt", "w", encoding="utf-8") as f:
    f.write(corpus_text)

# Statistiques avant / après
raw_pages = extract_pages(PDF_PATH)
raw_text = "\\n".join(raw_pages)
stats = {
    "": ["PDF brut", "Corpus nettoyé"],
    "Pages": [info["total_pages"], info["final_page_count"]],
    "Mots": [len(raw_text.split()), len(corpus_text.split())],
    "Caractères": [len(raw_text), len(corpus_text)],
}
stats_df = pd.DataFrame(stats).set_index("")
print(stats_df.to_string())
print(f"\\nCorpus sauvegardé -> data/CORPUS_RAG_reduit_100pages.txt ({len(corpus_text)} caractères)")
"""))

cells.append(code("""# Visualisation du cleaning (pages conservées vs retirées)
import matplotlib.pyplot as plt

removed_front = len(info["front_matter_removed"])
end_start = info["end_matter_start_page"]
removed_end = (info["total_pages"] - end_start + 1) if end_start else 0
kept = info["final_page_count"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(["Retiré\\n(front-matter)", "Retiré\\n(biblio/annexes)", "Conservé\\n(corpus final)"],
              [removed_front, removed_end, kept],
              color=["#e07a5f", "#f2cc8f", "#81b29a"])
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, str(int(b.get_height())),
            ha="center", fontweight="bold")
ax.set_ylabel("Nombre de pages")
ax.set_title(f"Cleaning du PDF : {info['total_pages']} pages -> {kept} pages (~{PAGE_BUDGET} visées)")
plt.tight_layout()
plt.savefig("docs/cleaning_pages.png", dpi=130)
plt.show()
"""))

cells.append(code("""# Aperçu du début et de la fin du corpus final (contrôle qualité)
print("=== DÉBUT DU CORPUS (600 premiers caractères) ===")
print(corpus_text[:600])
print("\\n=== FIN DU CORPUS (300 derniers caractères) ===")
print(corpus_text[-300:])
"""))

cells.append(md("""**Lecture du résultat** : le PDF brut (191 pages) est réduit à **100 pages** de contenu utile. Les pages retirées sont exactement celles interdites par les consignes (remerciements, sommaire, listes de figures/tableaux), la couverture d'archive HAL (boilerplate hors thèse) et le end-matter (bibliographie + annexes, inutiles pour un RAG). Le corpus commence donc directement par la page de titre de la thèse. Le fichier `data/CORPUS_RAG_reduit_100pages.txt` est le corpus officiel utilisé par **toutes** les comparaisons suivantes.

---
"""))

part1 = cells

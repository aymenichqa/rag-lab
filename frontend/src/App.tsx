import { useEffect, useState } from "react";
import ArchitectureView from "./ArchitectureView";
import QueryView from "./QueryView";
import StageView from "./StageView";
import WorkflowView from "./WorkflowView";
import { api, BestMethods } from "./api";

type StageKey = "chunking" | "embedding";
type View =
  | "workflow"
  | StageKey
  | "chunking-metrics"
  | "embedding-metrics"
  | "retrieval"
  | "retrieval-metrics"
  | "query"
  | "architecture";

const SUBJECT_BUTTONS: {
  key: View;
  num: string;
  label: string;
  sub: string;
}[] = [
  { key: "workflow", num: "01", label: "RAG Workflow", sub: "question → réponse, bout en bout" },
  { key: "chunking", num: "02", label: "Chunking", sub: "7 méthodes" },
  { key: "chunking-metrics", num: "03", label: "Affichage des métriques de Chunking", sub: "graphiques & tableaux" },
  { key: "embedding", num: "04", label: "Embedding", sub: "7 méthodes" },
  { key: "embedding-metrics", num: "05", label: "Affichage des métriques de chaque méthode", sub: "graphiques & tableaux" },
  { key: "retrieval", num: "06", label: "Retrieval", sub: "6 méthodes" },
  { key: "retrieval-metrics", num: "07", label: "Affichage des métriques de Retrieval", sub: "graphiques & tableaux" },
];

const SECONDARY_BUTTONS: {
  key: View;
  dot: string;
  label: string;
}[] = [
  { key: "retrieval", dot: "#7cc47c", label: "Recherche (Retrieval)" },
  { key: "query", dot: "#4ade80", label: "Questionnaire" },
  { key: "architecture", dot: "#d6ddd2", label: "Architecture" },
];

const STAGE_SUB: Record<View, string> = {
  workflow: "Le pipeline complet en une vue : question → chunking → embedding → retrieval → réponse générée.",
  chunking: "7 méthodes de découpage comparées sur 9 métriques — cohérence, redondance, qualité de retrieval.",
  "chunking-metrics": "7 méthodes de découpage comparées sur 9 métriques — cohérence, redondance, qualité de retrieval.",
  embedding: "7 méthodes de vectorisation comparées sur 8 métriques — du Word2Vec local au SBERT pré-entraîné.",
  "embedding-metrics": "7 méthodes de vectorisation comparées sur 8 métriques — du Word2Vec local au SBERT pré-entraîné.",
  retrieval: "6 stratégies de recherche comparées — de la similarité brute au re-ranking cross-encoder.",
  "retrieval-metrics": "6 stratégies de recherche comparées — de la similarité brute au re-ranking cross-encoder.",
  query: "Posez une question : le pipeline gagnant récupère les passages et un LLM local rédige la réponse avec sources.",
  architecture: "Comment le notebook, l'API FastAPI et l'interface React s'articulent — et le pipeline retenu.",
};

function isMetricsView(view: View): view is "chunking-metrics" | "embedding-metrics" {
  return view === "chunking-metrics" || view === "embedding-metrics";
}

function metricsStageKey(view: View): StageKey {
  return view === "chunking-metrics" ? "chunking" : "embedding";
}

export default function App() {
  const [view, setView] = useState<View>("workflow");
  const [best, setBest] = useState<BestMethods | null>(null);

  useEffect(() => {
    api.bestMethods().then(setBest).catch(() => setBest(null));
  }, []);

  const activeSubjectKey = SUBJECT_BUTTONS.find((b) => b.key === view)?.key ?? "workflow";

  return (
    <div className="app">
      <header className="hero">
        <span className="kicker">PFA · Stage d'été · 4ème année IIR · Pipeline RAG</span>
        <h1>RAG Lab — comparatif des méthodes à chaque étape</h1>
        <p className="sub">
          Trois étapes du pipeline (découpage, vectorisation, recherche) sont évaluées avec plusieurs méthodes et
          plusieurs métriques sur un corpus réel de 100 pages. Pour chaque étape, la meilleure méthode est désignée
          et justifiée ; la console interroge le corpus en live.
        </p>
      </header>
      <hr className="hero-rule" />

      {/* index des 4 boutons du sujet */}
      <div className="subject-band">
        {SUBJECT_BUTTONS.map((b) => (
          <button
            key={b.key}
            data-key={b.key}
            className={`subject-btn${view === b.key ? " active" : ""}`}
            onClick={() => setView(b.key)}
          >
            <span className="num">{b.num}</span>
            <span className="label">{b.label}</span>
            <span className="sub">{b.sub}</span>
          </button>
        ))}
      </div>

      {/* nav secondaire discrète : fond vert foncé, texte blanc */}
      <nav className="secondary-nav">
        <span className="sn-label">Naviguer</span>
        {SECONDARY_BUTTONS.map((b) => (
          <button
            key={b.key}
            className={`sec-btn${view === b.key ? " active" : ""}`}
            onClick={() => setView(b.key)}
          >
            <span className="sn-dot" style={{ background: b.dot }} />
            {b.label}
          </button>
        ))}
      </nav>

      <p className="stage-sub">{STAGE_SUB[view]}</p>

      {best && view !== "architecture" && view !== "query" && view !== "workflow" && (
        <p className="pipeline-line">
          Pipeline gagnant : <b>{best.pipeline_final.chunking}</b>
          <span className="arr">→</span>
          <b>{best.pipeline_final.embedding}</b>
          <span className="arr">→</span>
          <b>{best.pipeline_final.retrieval}</b>
        </p>
      )}

      {view === "workflow" ? (
        <WorkflowView onNavigate={(v) => setView(v as View)} />
      ) : view === "architecture" ? (
        <ArchitectureView />
      ) : view === "query" ? (
        <QueryView />
      ) : view === "retrieval" || view === "retrieval-metrics" ? (
        <StageView stage="retrieval" />
      ) : isMetricsView(view) ? (
        <StageView stage={metricsStageKey(view)} />
      ) : (
        <StageView stage={view} />
      )}

      <footer>
        Corpus : thèse académique (191 → 100 pages après cleaning). Backend FastAPI, frontend React + Recharts,
        LLM local Ollama avec repli Hugging Face. Les résultats affichés proviennent du notebook exécuté — source
        de vérité unique.
      </footer>
    </div>
  );
}

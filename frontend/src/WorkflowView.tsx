import { useEffect, useState } from "react";
import { api, BestMethods, QueryMethodKey, QueryResponse, StageKey, StageResults } from "./api";

/* Métriques « phares » par étape pour les mini-tableaux du workflow. */
const MINI_METRICS: Record<StageKey, RegExp[]> = {
  chunking: [/Similarité intra/i, /F1/i, /MRR/i, /Score composite/i],
  embedding: [/MRR/i, /n?dcg/i, /cosine/i, /Score composite/i],
  retrieval: [/n?dcg/i, /map/i, /mrr/i, /latence/i],
};

function pickMetrics(stage: StageKey, cols: string[]): string[] {
  const picked: string[] = [];
  for (const re of MINI_METRICS[stage]) {
    const found = cols.find((c) => re.test(c) && !picked.includes(c));
    if (found) picked.push(found);
  }
  return picked.slice(0, 4);
}

const STAGE_COLOR: Record<StageKey, string> = {
  chunking: "#1e5c42",
  embedding: "#44546a",
  retrieval: "#b45309",
};

const STAGE_TITLE: Record<StageKey, string> = {
  chunking: "Chunking",
  embedding: "Embedding",
  retrieval: "Retrieval",
};

function MiniTable({ stage, data }: { stage: StageKey; data: StageResults | null }) {
  if (!data) return <div className="mini-skeleton" />;
  const cols = pickMetrics(stage, data.metrics_columns);
  return (
    <table className="mini-table">
      <thead>
        <tr>
          <th>Méthode</th>
          {cols.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.ranking.map((name) => {
          const row = data.methods.find((m) => m.name === name)!;
          const isWinner = name === data.best.name;
          return (
            <tr key={name} className={isWinner ? "winner" : undefined}>
              <td>
                {isWinner && <span className="mini-flag">★ </span>}
                {name}
              </td>
              {cols.map((c) => (
                <td key={c}>
                  {typeof row[c] === "number" ? (row[c] as number).toFixed(2) : "—"}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function WorkflowView({ onNavigate }: { onNavigate: (v: string) => void }) {
  const [best, setBest] = useState<BestMethods | null>(null);
  const [stages, setStages] = useState<Record<StageKey, StageResults | null>>({
    chunking: null,
    embedding: null,
    retrieval: null,
  });
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState<QueryMethodKey>("all");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.bestMethods().then(setBest).catch(() => setBest(null));
    (["chunking", "embedding", "retrieval"] as StageKey[]).forEach((s) =>
      api
        .stageResults(s)
        .then((d) => setStages((prev) => ({ ...prev, [s]: d })))
        .catch(() => undefined),
    );
  }, []);

  const ask = async () => {
    if (question.trim().length < 3 || busy) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.query(question.trim(), method, 3));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="workflow">
      {/* 1. Query */}
      <section className="wf-query">
        <div className="wf-step-label">1 · Question (query)</div>
        <div className="wf-query-box">
          <input
            type="text"
            value={question}
            placeholder="Posez une question sur le corpus…"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <select value={method} onChange={(e) => setMethod(e.target.value as QueryMethodKey)}>
            <option value="all">Méthode gagnante</option>
            <option value="brute">Similarité brute</option>
            <option value="bm25">BM25</option>
            <option value="dense">Dense (DPR)</option>
            <option value="hybrid">Hybride</option>
            <option value="rerank">Re-ranking</option>
          </select>
          <button className="wf-go" onClick={ask} disabled={busy}>
            {busy ? "Recherche…" : "Interroger"}
          </button>
        </div>
        {error && <div className="error-box" style={{ marginTop: 12 }}>Erreur : {error}</div>}
      </section>

      {/* 2. Les trois étapes avec leurs métriques */}
      <div className="wf-stages">
        {(["chunking", "embedding", "retrieval"] as StageKey[]).map((s, i) => (
          <section className="wf-stage" key={s}>
            <div className="wf-step-label">
              {["2", "3", "4"][i]} · {STAGE_TITLE[s]}
            </div>
            <div className="wf-card" style={{ borderTopColor: STAGE_COLOR[s] }}>
              <h4 style={{ color: STAGE_COLOR[s] }}>{STAGE_TITLE[s]}</h4>
              {best && (
                <p className="wf-best">
                  Meilleure : <b>{best[s].name}</b>
                </p>
              )}
              <MiniTable stage={s} data={stages[s]} />
              <button className="wf-more" onClick={() => onNavigate(s === "retrieval" ? "retrieval" : s)}>
                Détails & graphiques →
              </button>
            </div>
          </section>
        ))}
      </div>

      {/* 3. Passages récupérés */}
      {result && (
        <section className="wf-passages">
          <div className="wf-step-label">5 · Passages récupérés ({result.method})</div>
          {result.passages.map((p) => (
            <div className="wf-passage" key={p.index}>
              <span className="wf-rank">#{p.rank}</span>
              <span className="wf-chunk mono">chunk {p.index}</span>
              <p>{p.text}</p>
            </div>
          ))}
        </section>
      )}

      {/* 4. Réponse générée */}
      {result?.generation && (
        <section className="wf-answer">
          <div className="wf-step-label">6 · Réponse générée — {result.generation.backend}</div>
          <div className="wf-answer-box">
            <p>{result.generation.answer}</p>
          </div>
        </section>
      )}
    </div>
  );
}

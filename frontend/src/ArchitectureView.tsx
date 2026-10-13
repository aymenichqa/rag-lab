import { useEffect, useState } from "react";
import { api, Architecture, LlmBenchmark } from "./api";

export default function ArchitectureView() {
  const [arch, setArch] = useState<Architecture | null>(null);
  const [bench, setBench] = useState<LlmBenchmark | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([api.architecture(), api.llmBenchmark().catch(() => null)])
      .then(([a, b]) => {
        if (alive) {
          setArch(a);
          if (b) setBench(b);
        }
      })
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <div className="error-box">Erreur : {error}</div>;
  if (!arch) return <div className="skeleton" />;

  const llm = arch.stack.llm as { model?: string; ollama_ready?: boolean } | undefined;

  return (
    <div>
      <section className="podium">
        <div className="label">Pipeline retenu (méthodes gagnantes)</div>
        <p className="pipeline-line" style={{ marginTop: 8 }}>
          <b>Découpage :</b> {arch.pipeline.chunking}
          <span className="arr">→</span>
          <b>Vectorisation :</b> {arch.pipeline.embedding}
          <span className="arr">→</span>
          <b>Recherche :</b> {arch.pipeline.retrieval}
        </p>
        <p className="why" style={{ marginTop: 12 }}>
          {String(arch.corpus.n_chunks ?? "—")} chunks · vecteurs de {String(arch.corpus.embedding_dim ?? "—")}{" "}
          dimensions · corpus de{" "}
          {typeof arch.corpus.corpus_chars === "number" ? arch.corpus.corpus_chars.toLocaleString("fr-FR") : "—"}{" "}
          caractères.
        </p>
      </section>

      <div className="arch-grid">
        <section className="panel arch-col">
          <h4>Offline — notebook (source de vérité)</h4>
          <ul>
            {arch.offline.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </section>
        <section className="panel arch-col">
          <h4>Online — application web</h4>
          <ul>
            {arch.online.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
          <div className="stack-line">
            {Object.entries(arch.stack)
              .filter(([, v]) => typeof v === "string")
              .map(([k, v]) => (
                <span className="chip" key={k}>
                  <b>{k}</b> {v as string}
                </span>
              ))}
          </div>
        </section>
      </div>

      {bench && bench.models.length > 0 && (
        <section className="panel" style={{ marginTop: 18 }}>
          <h4 style={{ marginBottom: 8 }}>Infrastructure LLM — benchmark des modèles locaux</h4>
          <table className="data">
            <thead>
              <tr>
                <th>Modèle</th>
                <th>Latence moy. (s)</th>
                <th>ms / token</th>
                <th>Citations moy. / question</th>
              </tr>
            </thead>
            <tbody>
              {bench.models.map((m) => (
                <tr key={m.model}>
                  <td>{m.model}</td>
                  <td>{m.latence_moy_s ?? "—"}</td>
                  <td>{m.ms_par_token ?? "—"}</td>
                  <td>{m.citations_moy ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {llm && (
            <p className="hint" style={{ marginTop: 10 }}>
              Modèle servi en local via Ollama{llm.model ? ` (${llm.model})` : ""} — repli automatique sur un petit
              modèle Hugging Face si Ollama est indisponible.
            </p>
          )}
        </section>
      )}

      <section className="panel">
        <h4 style={{ marginBottom: 10 }}>Schéma d'architecture (généré par le notebook)</h4>
        <img className="arch-img" src="/architecture.png?v=2" alt="Architecture du projet RAG" />
      </section>
    </div>
  );
}

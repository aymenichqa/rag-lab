import { useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, QUERY_METHODS, QueryMethodKey, QueryResponse } from "./api";

const STOP = new Set([
  "comment", "quel", "quelle", "quels", "quelles", "est", "dans", "pour", "avec", "pourquoi",
  "rôle", "role", "impact", "effet", "facteurs", "outils", "qu'est", "les", "des", "une",
  "sont", "cette", "leur", "plus", "mais", "que", "qui", "par", "sur", "entre",
]);

function questionTokens(q: string): string[] {
  return q
    .toLowerCase()
    .split(/[^a-zàâäéèêëîïôöùûüç]+/)
    .filter((w) => w.length >= 4 && !STOP.has(w));
}

function Highlight({ text, tokens }: { text: string; tokens: string[] }) {
  if (!tokens.length) return <>{text}</>;
  const esc = tokens.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  // \b aux deux bouts : on surligne le mot entier, jamais un fragment à l'intérieur d'un autre mot
  const splitRe = new RegExp(`(\\b(?:${esc})\\b)`, "gi");
  // test sans flag /g (pas d'état lastIndex) : une partie est surlignée si elle EST un token entier
  const isToken = new RegExp(`^(?:${esc})$`, "i");
  const parts = text.split(splitRe);
  return (
    <>
      {parts.map((p, i) =>
        p && isToken.test(p) ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>,
      )}
    </>
  );
}

export default function QueryView() {
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState<QueryMethodKey>("all");
  const [topK, setTopK] = useState(4);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ask = async () => {
    if (question.trim().length < 3 || busy) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.query(question.trim(), method, topK));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const tokens = questionTokens(question);

  return (
    <div>
      <section className="panel">
        <h3>Interroger le corpus</h3>
        <p className="hint">
          Le pipeline gagnant (chunking → embedding → retrieval) tourne en live sur le corpus de la thèse ; la réponse
          est générée par un LLM local (Ollama, repli Hugging Face) uniquement à partir des passages récupérés.
        </p>
        <div className="query-bar">
          <input
            type="text"
            value={question}
            placeholder="Ex. : Comment les zones humides réduisent-elles la pollution par les nitrates ?"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <select value={method} onChange={(e) => setMethod(e.target.value as QueryMethodKey)}>
            {QUERY_METHODS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[3, 4, 5, 6].map((k) => (
              <option key={k} value={k}>
                Top {k}
              </option>
            ))}
          </select>
          <button className="go" onClick={ask} disabled={busy}>
            {busy ? "Recherche…" : "Interroger"}
          </button>
        </div>
        {error && <div className="error-box" style={{ marginTop: 12 }}>Erreur : {error}</div>}
      </section>

      {result && (
        <>
          {result.generation && (
            <section className="answer">
              <div className="backend">Réponse générée — {result.generation.backend}</div>
              <p>{result.generation.answer}</p>
            </section>
          )}

          <section className="panel" style={{ marginTop: 18 }}>
            <h3>Passages récupérés ({result.method})</h3>
            <p className="hint">Les termes de la question sont surlignés dans les extraits du corpus.</p>
            {result.passages.map((p) => (
              <div className="passage" key={p.index}>
                <div className="meta">
                  <span className="rank">#{p.rank}</span>
                  <span className="m mono">chunk {p.index}</span>
                  <span className="m">{p.method}</span>
                </div>
                <p>
                  <Highlight text={p.text} tokens={tokens} />
                </p>
              </div>
            ))}
          </section>

          {result.timings_ms.length > 1 && (
            <section className="panel" style={{ marginTop: 18 }}>
              <h3>Leaderboard live — latence par méthode</h3>
              <p className="hint">Chronométrage réel des 6 méthodes de retrieval sur cette question.</p>
              <div className="chart-box" style={{ height: 260 }}>
                <ResponsiveContainer>
                  <BarChart data={[...result.timings_ms].sort((a, b) => a.ms - b.ms)}>
                    <XAxis dataKey="method" tick={{ fill: "#6f6a5c", fontSize: 10.5 }} interval={0} angle={-16}
                           dy={8} height={64} />
                    <YAxis tick={{ fill: "#6f6a5c", fontSize: 11 }} unit=" ms" />
                    <Tooltip
                      contentStyle={{ background: "#faf9f4", border: "1px solid #d9d4c5", borderRadius: 0 }}
                      formatter={(v) => [`${v} ms`, "latence"]}
                    />
                    <Bar dataKey="ms" radius={[0, 0, 0, 0]} animationDuration={600}>
                      {[...result.timings_ms].sort((a, b) => a.ms - b.ms).map((t, i) => (
                        <Cell key={t.method} fill={i === 0 ? "#1e5c42" : "#44546a"} fillOpacity={0.85} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, PolarAngleAxis, PolarGrid,
  PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, StageKey, StageResults } from "./api";

const ACCENTS: Record<StageKey, string> = {
  chunking: "#b45309",
  embedding: "#1e5c42",
  retrieval: "#44546a",
};

const STAGE_LABEL: Record<StageKey, string> = {
  chunking: "Découpage (chunking)",
  embedding: "Vectorisation (embedding)",
  retrieval: "Recherche (retrieval)",
};

// Métriques « notables » par étape : colonnes de chiffres → nom court + suffixe % pour les % vs médiane
const PCT_SUFFIX = " %";

function shortName(name: string): string {
  const m = name.match(/^\d+\.\s*(.*)$/);
  return m ? m[1] : name;
}

type Row = { name: string; value: number; raw: number | string | null; winner: boolean };

export default function StageView({ stage }: { stage: StageKey }) {
  const [data, setData] = useState<StageResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<string | null>(null);
  const accent = ACCENTS[stage];

  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    api
      .stageResults(stage)
      .then((d) => {
        if (!alive) return;
        setData(d);
        // métrique par défaut : première métrique binaire (booléen non null) avec des valeurs
        const first = d.metrics_columns.find((c) => d.higher_is_better[c] !== null);
        setMetric(first ?? d.metrics_columns[0]);
      })
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [stage]);

  const baseMetrics = useMemo(() => {
    if (!data) return [];
    return data.metrics_columns.filter((c) => {
      if (c.endsWith(PCT_SUFFIX)) return false;
      return data.methods.some((m) => typeof m[c] === "number");
    });
  }, [data]);

  const metricRows: Row[] = useMemo(() => {
    if (!data || !metric) return [];
    return data.methods.map((m) => ({
      name: shortName(m.name),
      value: typeof m[metric] === "number" ? (m[metric] as number) : 0,
      raw: m[metric],
      winner: m.name === data.best.name,
    }));
  }, [data, metric]);

  // Tri une seule fois : le graphique ET les couleurs des barres utilisent le même ordre.
  const sortedRows: Row[] = useMemo(
    () => [...metricRows].sort((a, b) => (b.value as number) - (a.value as number)),
    [metricRows],
  );

  const radarData = useMemo(() => {
    if (!data) return [];
    // normalise chaque métrique numérique sur 0-1 pour comparer toutes les méthodes sur un seul radar
    return baseMetrics.map((c) => {
      const vals = data.methods.map((m) => (typeof m[c] === "number" ? (m[c] as number) : null));
      const nums = vals.filter((v): v is number => v !== null);
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const span = max - min || 1;
      const entry: Record<string, number | string> = { metric: c };
      for (const m of data.methods) {
        const v = m[c];
        entry[shortName(m.name)] =
          typeof v === "number" ? +(((v - min) / span) * (data.higher_is_better[c] === false ? -1 : 1)).toFixed(3) : 0;
      }
      return entry;
    });
  }, [data, baseMetrics]);

  if (error) return <div className="error-box">Erreur : {error}</div>;
  if (!data) {
    return (
      <>
        <div className="skeleton" />
        <div className="skeleton" />
      </>
    );
  }

  const podiumMethods = data.ranking.slice(0, 3).map(shortName);

  return (
    <div>
      {/* podium */}
      <section className="podium">
        <div className="label">Meilleure méthode — {STAGE_LABEL[stage]}</div>
        <div className="name">
          <span className="medal">★</span>
          {data.best.name}
        </div>
        <p className="why">{data.best.justification}</p>
        <div className="chips">
          {podiumMethods.map((p, i) => (
            <span className="chip" key={p}>
              {["🥇", "🥈", "🥉"][i]} <b>{p}</b>
            </span>
          ))}
        </div>
      </section>

      {/* graphique principal : barres par métrique sélectionnable */}
      <section className="panel">
        <h3>Comparaison par métrique</h3>
        <p className="hint">
          Cliquez sur une métrique pour mettre à jour le graphique. Les barres sont triées de la meilleure à la moins
          bonne pour la métrique choisie.
        </p>
        <div className="metric-tabs">
          {baseMetrics.map((c) => (
            <button key={c} className={metric === c ? "on" : ""} onClick={() => setMetric(c)}
                    style={metric === c ? { color: accent } : undefined}>
              {c}
            </button>
          ))}
        </div>
        <div className="chart-box">
          <ResponsiveContainer>
            <BarChart data={sortedRows}>
              <CartesianGrid stroke="#e8e4d8" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: "#6f6a5c", fontSize: 11 }} interval={0} angle={-18}
                     dy={8} height={58} />
              <YAxis tick={{ fill: "#6f6a5c", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#faf9f4", border: "1px solid #d9d4c5", borderRadius: 0 }}
                labelStyle={{ color: "#1c1b17" }}
                formatter={(v) => [typeof v === "number" ? +v.toFixed(3) : v, metric ?? ""]}
              />
              <Bar dataKey="value" radius={[0, 0, 0, 0]} animationDuration={700}>
                {sortedRows.map((r) => (
                  <Cell key={r.name} fill={r.winner ? "#1e5c42" : accent} fillOpacity={r.winner ? 1 : 0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* radar + tableau */}
      <div className="two-col">
        <section className="panel">
          <h3>Profil comparatif (radar normalisé)</h3>
          <p className="hint">
            Chaque métrique est normalisée sur 0-1 (orientée « plus c'est mieux ») pour comparer le profil global des
            méthodes.
          </p>
          <div className="chart-box tall">
            <ResponsiveContainer>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#d9d4c5" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "#6f6a5c", fontSize: 10.5 }} />
                <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 1]} />
                {data.methods.map((m, i) => (
                  <Radar
                    key={m.name}
                    name={shortName(m.name)}
                    dataKey={shortName(m.name)}
                    stroke={m.name === data.best.name ? "#1e5c42" : ACCENTS[stage]}
                    fill={m.name === data.best.name ? "#1e5c42" : ACCENTS[stage]}
                    fillOpacity={m.name === data.best.name ? 0.18 : 0.05}
                    strokeOpacity={m.name === data.best.name ? 1 : 0.45}
                    strokeWidth={m.name === data.best.name ? 2 : 1}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: 11.5, color: "#6f6a5c" }} />
                <Tooltip contentStyle={{ background: "#faf9f4", border: "1px solid #d9d4c5", borderRadius: 0 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel">
          <h3>Tableau détaillé</h3>
          <p className="hint">Chiffres exacts pour le mémoire — la ligne du gagnant est surlignée.</p>
          <details className="table-wrap" open>
            <summary>Afficher / masquer le tableau complet</summary>
            <div style={{ overflowX: "auto" }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>Méthode</th>
                    {baseMetrics.map((c) => (
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
                          {name}
                          {isWinner && <span className="best-flag">★ gagnant</span>}
                        </td>
                        {baseMetrics.map((c) => (
                          <td key={c}>
                            {typeof row[c] === "number" ? (row[c] as number).toFixed(3) : String(row[c] ?? "—")}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      </div>
    </div>
  );
}

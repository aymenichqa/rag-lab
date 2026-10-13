// Client API typé — communication REST avec le backend FastAPI.

export type StageKey = "chunking" | "embedding" | "retrieval";

export type MethodResult = {
  name: string;
  [metric: string]: number | string | null;
};

export type StageResults = {
  stage: StageKey;
  metrics_columns: string[];
  higher_is_better: Record<string, boolean | null>;
  methods: MethodResult[];
  best: { name: string; justification: string };
  ranking: string[];
};

export type BestMethods = {
  chunking: StageBest;
  embedding: StageBest;
  retrieval: StageBest;
  pipeline_final: { chunking: string; embedding: string; retrieval: string; resume: string };
};

export type StageBest = { name: string; justification: string; score: number };

export type LlmBenchmark = { models: LlmBenchRow[] };
export type LlmBenchRow = {
  model: string;
  latence_moy_s: number;
  ms_par_token: number;
  citations_moy: number;
};

export type Architecture = {
  offline: string[];
  online: string[];
  stack: Record<string, unknown>;
  pipeline: { chunking: string; embedding: string; retrieval: string };
  corpus: Record<string, unknown>;
};

export type QueryResponse = {
  question: string;
  method: string;
  passages: { rank: number; index: number; text: string; method: string }[];
  timings_ms: { method: string; ms: number }[];
  generation: { backend: string; answer: string; sources: string[] } | null;
};

export type QueryMethodKey =
  | "all"
  | "brute"
  | "ann"
  | "bm25"
  | "dense"
  | "hybrid"
  | "rerank";

export const QUERY_METHODS: { key: QueryMethodKey; label: string }[] = [
  { key: "all", label: "Toutes (leaderboard)" },
  { key: "brute", label: "Similarité brute" },
  { key: "ann", label: "ANN (FAISS)" },
  { key: "bm25", label: "Mots-clés (BM25)" },
  { key: "dense", label: "Dense (DPR)" },
  { key: "hybrid", label: "Hybride (RRF)" },
  { key: "rerank", label: "Re-ranking" },
];

const BASE = "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`GET ${path} : ${r.status}`);
  return r.json();
}

export const api = {
  stageResults: (stage: StageKey) => get<StageResults>(`/api/results/${stage}`),
  bestMethods: () => get<BestMethods>("/api/best-methods"),
  llmBenchmark: () => get<LlmBenchmark>("/api/llm-benchmark"),
  architecture: () => get<Architecture>("/api/architecture"),
  query: async (question: string, method: QueryMethodKey, topK: number): Promise<QueryResponse> => {
    const r = await fetch(`${BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, method, top_k: topK }),
    });
    if (!r.ok) throw new Error(`POST /api/query : ${r.status}`);
    return r.json();
  },
};

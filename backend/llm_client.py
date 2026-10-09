# -*- coding: utf-8 -*-
"""Client LLM : génération de la réponse via Ollama (repli Hugging Face)."""
import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "llama3.2:1b")

_hf_pipeline = None


def _ollama_models() -> list[str]:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


def _generate_ollama(prompt: str, model: str, max_tokens: int = 320) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]


def _generate_hf(prompt: str, max_new_tokens: int = 250) -> str:
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline
        _hf_pipeline = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype="auto",
        )
    messages = [{"role": "user", "content": prompt}]
    out = _hf_pipeline(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"][-1]["content"]


def build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[Source {i+1}] {p[:600]}" for i, p in enumerate(passages))
    return (
        "Tu es un assistant documentaire. Réponds en français, de façon factuelle et concise "
        "(5 phrases maximum), UNIQUEMENT à partir des extraits ci-dessous. Cite les sources "
        "utilisées sous la forme [Source N]. Si les extraits ne permettent pas de répondre, "
        f"dis-le clairement.\n\n### Extraits du corpus\n{context}\n\n### Question\n{question}\n\n### Réponse :"
    )


def generate_with_context(question: str, passages: list[str]) -> dict:
    """Retourne {backend, answer, sources} — n'échoue jamais silencieusement."""
    prompt = build_prompt(question, passages)
    models = _ollama_models()
    if any(m.startswith("llama3.2") for m in models):
        try:
            return {
                "backend": f"Ollama ({GEN_MODEL})",
                "answer": _generate_ollama(prompt, GEN_MODEL).strip(),
                "sources": [f"[Source {i+1}] {p[:120]}..." for i, p in enumerate(passages)],
            }
        except Exception:
            pass
    answer = _generate_hf(prompt)
    return {
        "backend": "Hugging Face (Qwen2.5-0.5B-Instruct)",
        "answer": answer.strip(),
        "sources": [f"[Source {i+1}] {p[:120]}..." for i, p in enumerate(passages)],
    }

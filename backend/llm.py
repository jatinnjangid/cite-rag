"""
llm.py — LLM abstraction supporting Gemini and OpenAI.

Select the provider via LLM_PROVIDER env var ("gemini" | "openai").
The grounded-answer prompt is defined here and shared by both providers:
  → Answer ONLY from the supplied context.
  → If the answer is not in the context, say exactly "not found in documents".
"""

from typing import List, Dict, Any

from config import cfg


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise document question-answering assistant.
Answer the user's question using ONLY the context passages provided below.
Rules:
- If the answer is present in the context, give a concise, accurate answer.
- Cite which document and page you drew information from.
- If the context does not contain enough information to answer, respond with
  exactly: "not found in documents"
- Do NOT use any outside knowledge. Do NOT make up information.
"""

def _build_user_message(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    context_lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_lines.append(
            f"[{i}] Source: {chunk['doc_name']}, Page {chunk['page_num']}\n"
            f"{chunk['text']}"
        )
    context_block = "\n\n".join(context_lines)
    return f"Context:\n{context_block}\n\nQuestion: {question}"


# ── Provider implementations ───────────────────────────────────────────────────

def _call_gemini(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Call Google Gemini via the google-generativeai SDK."""
    import google.generativeai as genai

    if not cfg.gemini_api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    genai.configure(api_key=cfg.gemini_api_key)
    model = genai.GenerativeModel(
        model_name=cfg.gemini_model,
        system_instruction=_SYSTEM_PROMPT,
    )

    user_message = _build_user_message(question, context_chunks)
    response = model.generate_content(user_message)
    return response.text.strip()


def _call_openai(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Call OpenAI Chat Completions API."""
    from openai import OpenAI

    if not cfg.openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    client = OpenAI(api_key=cfg.openai_api_key)
    user_message = _build_user_message(question, context_chunks)

    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,   # deterministic — we want factual grounded answers
    )
    return response.choices[0].message.content.strip()


# ── Public interface ───────────────────────────────────────────────────────────

def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a grounded answer using the configured LLM provider.

    Args:
        question:       The user's question.
        context_chunks: Retrieved chunks (each with doc_name, page_num, text).

    Returns:
        Answer string. May be "not found in documents" if context is empty
        or the LLM decides the answer isn't present.

    Raises:
        ValueError:       If LLM_PROVIDER is not "gemini" or "openai".
        EnvironmentError: If the required API key is missing.
    """
    if not context_chunks:
        return "not found in documents"

    provider = cfg.llm_provider.lower()
    if provider == "gemini":
        return _call_gemini(question, context_chunks)
    elif provider == "openai":
        return _call_openai(question, context_chunks)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{cfg.llm_provider}'. "
            "Set it to 'gemini' or 'openai' in your .env file."
        )

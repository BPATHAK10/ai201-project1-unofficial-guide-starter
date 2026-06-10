"""
Grounded answer generation for The Unofficial Guide.

Pipeline: retrieve top-k chunks -> build a context-only prompt -> ask Groq's
llama-3.3-70b-versatile to answer using ONLY that context -> return the answer
plus the list of source documents it was drawn from.

Grounding is enforced two ways:
  1. The system prompt tells the model to answer only from the provided context
     and to refuse ("I don't have enough information on that.") otherwise.
  2. The source list is built in code from the retrieved chunks' metadata, not
     parsed out of the model's text, so attribution can't be hallucinated.

Usage:
    from query import ask
    result = ask("How much do the co-ops cost?")
    print(result["answer"], result["sources"])
"""

import os

from dotenv import load_dotenv
from groq import Groq

from embed import retrieve

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = f"""You are The Unofficial Guide, a question-answering assistant about \
off-campus housing at UC Berkeley. You answer using ONLY the numbered context \
documents provided in each question.

Rules:
- Use only information found in the context. Do not add facts from your own \
general knowledge, even if you are confident they are true.
- If the context does not contain enough information to answer, reply with \
exactly this sentence and nothing else: "{REFUSAL}"
- Cite the context documents you used with their bracket numbers, e.g. [1], [3].
- Keep answers concise and specific. Quote figures (rent, hours, dates) exactly \
as they appear in the context."""

_client = None


def _groq():
    """Create the Groq client once, reading GROQ_API_KEY from the environment."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def _number_sources(hits):
    """Assign each unique source document a citation number (by first
    appearance). Returns (sources, source_to_num) where sources is a list of
    {n, source, url} dicts."""
    source_to_num, sources = {}, []
    for hit in hits:
        if hit["source"] not in source_to_num:
            n = len(sources) + 1
            source_to_num[hit["source"]] = n
            sources.append({"n": n, "source": hit["source"], "url": hit["url"]})
    return sources, source_to_num


def _format_context(hits, source_to_num):
    """Label each chunk with its source's citation number, so the model's
    [n] citations line up with the numbered Sources list. Chunks from the same
    document share a number."""
    blocks = []
    for hit in hits:
        n = source_to_num[hit["source"]]
        blocks.append(f"[{n}] (source: {hit['source']})\n{hit['text']}")
    return "\n\n".join(blocks)


def ask(question, k=5):
    """Answer a question grounded in retrieved context.

    Returns {answer, sources, hits} where sources is a list of
    {n, source, url} dicts built from the retrieved chunks (empty if the model
    declined to answer). Citation numbers are per document, so an answer's
    inline [n] matches the numbered source list.
    """
    hits = retrieve(question, k=k)
    sources, source_to_num = _number_sources(hits)
    context = _format_context(hits, source_to_num)

    response = _groq().chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context documents:\n\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content.strip()

    # Suppress source attribution when the model declined to answer.
    if REFUSAL.lower() in answer.lower():
        sources = []

    return {"answer": answer, "sources": sources, "hits": hits}


if __name__ == "__main__":
    demo_questions = [
        "How many work-shift hours per week do co-op house members do?",
        "What problems have tenants reported at Evans Manor?",
        "When should a student sign a lease for an August move-in?",
        # out-of-scope: nothing in the documents covers dining hall food
        "Which dining hall has the best food at UC Berkeley?",
    ]
    for q in demo_questions:
        result = ask(q)
        print("\n" + "=" * 75)
        print(f"Q: {q}")
        print("-" * 75)
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  [{s['n']}] {s['source']}")

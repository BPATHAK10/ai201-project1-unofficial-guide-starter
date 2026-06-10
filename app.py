"""
Chat interface for The Unofficial Guide (ChatGPT/Claude-style).

Run the index build first (python embed.py), then:
    python app.py
and open http://localhost:7860
"""

import gradio as gr

from query import ask

EXAMPLES = [
    "How much does a Berkeley co-op cost, and what's included?",
    "What problems have tenants reported at Evans Manor?",
    "Which Berkeley neighborhood is best for an engineering major?",
    "When should I start looking for an apartment for the fall?",
]


def respond(message, history):
    """Answer one question. history is ignored (each question is independent)."""
    result = ask(message)
    answer = result["answer"]
    if result["sources"]:
        links = "\n".join(
            f"{s['n']}. [{s['source']}]({s['url']})" if s["url"]
            else f"{s['n']}. {s['source']}"
            for s in result["sources"]
        )
        answer += f"\n\n---\n**Sources**\n{links}"
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="🏠 The Unofficial Guide",
    description=(
        "Ask about **off-campus housing at UC Berkeley** — co-ops, neighborhoods, "
        "rent, leasing timelines, tenant rights. Answers come only from collected "
        "student and community documents, with sources cited."
    ),
    examples=EXAMPLES,
)


if __name__ == "__main__":
    demo.launch()

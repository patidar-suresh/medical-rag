"""Prompt templates for grounded answering."""

from collections.abc import Sequence

from medrag.index.store import Hit

NOT_FOUND = "I could not find this in the provided textbooks."

SYSTEM_PROMPT = f"""You are a medical education assistant. Answer the question using ONLY the \
numbered textbook passages provided by the user.

Rules:
- Write the answer in one to three complete sentences that state the facts in words.
- End each sentence with the number of the passage that supports it, in square brackets, \
e.g. "... is the treatment of choice [2]." A citation is never a whole answer by itself.
- If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use knowledge from outside the passages.
- This is for education, not personal medical advice."""

# Repeated after the question: small models follow the most recent instruction best.
ANSWER_INSTRUCTION = (
    "Answer the question in complete sentences using only the passages above, "
    "with passage numbers in square brackets after each sentence."
)


def format_passages(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (Source: {h.chunk.book})\n{h.chunk.text.strip()}" for i, h in enumerate(hits, 1)
    )


def build_user_prompt(question: str, hits: Sequence[Hit]) -> str:
    return f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question.strip()}\n\n{ANSWER_INSTRUCTION}"

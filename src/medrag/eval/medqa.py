"""MedQA (USMLE-style, 4 options): accuracy without vs. with retrieval.

The MedQA questions were written alongside this exact textbook corpus, so the
RAG-vs-closed-book gap is a direct measure of how much retrieval helps.
Dataset: GBaker/MedQA-USMLE-4-options on Hugging Face (CC-BY-4.0), stored under data/.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrag.eval.common import read_jsonl
from medrag.generate.llm import Completion, LLMClient, Message
from medrag.generate.prompts import format_passages
from medrag.index.store import Hit

REPO_ID = "GBaker/MedQA-USMLE-4-options"
TEST_FILE = "phrases_no_exclude_test.jsonl"
LETTERS = ["A", "B", "C", "D"]

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": LETTERS}},
    "required": ["answer"],
}

SYSTEM = (
    "You are taking a medical licensing exam. Choose the single best answer. "
    'Reply as JSON: {"answer": "<letter>"}.'
)


@dataclass(frozen=True)
class MedQAItem:
    id: str
    question: str
    options: dict[str, str]
    answer: str  # gold letter
    step: str


def fetch_medqa(dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, TEST_FILE, repo_type="dataset", local_dir=dest_dir)
    return Path(path)


def load_medqa(path: Path, limit: int | None = None, seed: int = 7) -> list[MedQAItem]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `medrag eval fetch-medqa` first.")
    rows = read_jsonl(path)
    items = [
        MedQAItem(f"medqa-{i:04d}", r["question"], r["options"], r["answer_idx"], r.get("meta_info", ""))
        for i, r in enumerate(rows)
    ]
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items


def format_question(item: MedQAItem) -> str:
    opts = "\n".join(f"{k}. {item.options[k]}" for k in LETTERS if k in item.options)
    return f"{item.question}\n\n{opts}"


def ask(llm: LLMClient, item: MedQAItem, hits: list[Hit] | None) -> tuple[str | None, Completion]:
    body = format_question(item)
    if hits:
        body = (
            f"Reference passages from medical textbooks (use them if relevant):\n\n{format_passages(hits)}"
            f"\n\nQuestion:\n{body}"
        )
    out = llm.generate([Message("system", SYSTEM), Message("user", body)], json_schema=ANSWER_SCHEMA)
    try:
        letter = json.loads(out.text).get("answer")
    except json.JSONDecodeError:
        letter = None
    return (letter if letter in LETTERS else None), out

"""Online question answering: retrieve passages, then generate a grounded answer."""

import time
from collections.abc import Iterator
from dataclasses import dataclass

from medrag.generate.llm import Completion, LLMClient, Message
from medrag.generate.prompts import SYSTEM_PROMPT, build_user_prompt
from medrag.index.embedder import Embedder
from medrag.index.store import Hit, QdrantStore


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    hits: list[Hit]
    completion: Completion
    seconds_retrieval: float
    seconds_generation: float


class RAGPipeline:
    def __init__(self, embedder: Embedder, store: QdrantStore, llm: LLMClient, top_k: int) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(question), k or self.top_k)

    @staticmethod
    def messages(question: str, hits: list[Hit]) -> list[Message]:
        return [Message("system", SYSTEM_PROMPT), Message("user", build_user_prompt(question, hits))]

    def answer(self, question: str, k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, k)
        t1 = time.perf_counter()
        completion = self.llm.generate(self.messages(question, hits))
        t2 = time.perf_counter()
        return Answer(question, completion.text.strip(), hits, completion, t1 - t0, t2 - t1)

    def stream_answer(self, question: str, k: int | None = None) -> tuple[list[Hit], Iterator[str]]:
        hits = self.retrieve(question, k)
        return hits, self.llm.stream(self.messages(question, hits))

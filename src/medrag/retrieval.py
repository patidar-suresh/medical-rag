"""First-stage search (dense, BM25 or both), fusion, reranking, small-to-big and diversity.

query ─► [abbreviation expansion] ─► dense ┐
                                    BM25  ┴► RRF ─► [cross-encoder rerank]
      ─► parents (small-to-big) ─► [per-book cap] ─► top k
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from medrag.config import Settings
from medrag.index.embedder import Embedder
from medrag.index.reranker import Reranker
from medrag.index.sparse import BM25Encoder
from medrag.index.store import Hit, QdrantStore
from medrag.query import expand_abbreviations

Mode = Literal["dense", "sparse", "hybrid"]

# Parent-child: children fetched per wanted parent, since several children share a parent.
CHILDREN_PER_PARENT = 4


@dataclass(frozen=True)
class RetrievalOptions:
    mode: Mode = "dense"
    parent_mode: bool = False
    candidates: int = 50  # per first-stage list when fusing or reranking
    rrf_k: int = 60
    max_per_book: int = 0  # 0 = no cap
    expand_abbreviations: bool = False

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalOptions":
        return cls(
            mode=s.retrieval_mode,
            parent_mode=s.parent_chunk_size > 0,
            candidates=s.retrieval_candidates,
            rrf_k=s.rrf_k,
            max_per_book=s.max_per_book,
            expand_abbreviations=s.expand_abbreviations,
        )


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).

    Uses ranks only, so dense cosine scores and BM25 scores (different scales) never need
    normalising. A chunk ranked well by both retrievers beats one ranked first by only one.
    """
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    for hits in rankings:
        for rank, h in enumerate(hits, 1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first.setdefault(cid, h)
    order = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [Hit(first[c].chunk, scores[c]) for c in order]


def to_parents(hits: Sequence[Hit]) -> list[Hit]:
    """Small-to-big: each distinct parent once, at its best child's position."""
    seen: dict[str, Hit] = {}
    for h in hits:
        parent = h.chunk.as_parent()
        if parent.chunk_id not in seen:
            seen[parent.chunk_id] = Hit(parent, h.score)
    return list(seen.values())


def cap_per_book(hits: Sequence[Hit], cap: int, k: int) -> list[Hit]:
    """At most `cap` hits per book in the top k, so one large book can't crowd out the rest.

    Soft cap: if too few books are present to fill k, the best skipped hits fill the gap.
    """
    if cap <= 0:
        return list(hits[:k])
    kept: list[int] = []
    skipped: list[int] = []
    per_book: dict[str, int] = {}
    for i, h in enumerate(hits):
        if len(kept) == k:
            break
        if per_book.get(h.chunk.book, 0) < cap:
            kept.append(i)
            per_book[h.chunk.book] = per_book.get(h.chunk.book, 0) + 1
        else:
            skipped.append(i)
    if len(kept) < k:
        kept = sorted(kept + skipped[: k - len(kept)])
    return [hits[i] for i in kept]


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: QdrantStore,
        options: RetrievalOptions | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.options = options or RetrievalOptions()
        self.reranker = reranker
        self.sparse = BM25Encoder()  # query side needs no corpus statistics

    def _n_children(self, k: int) -> int:
        n = k * CHILDREN_PER_PARENT if self.options.parent_mode else k
        if self.options.mode != "dense" or self.reranker is not None:
            n = max(n, self.options.candidates)
        return n

    def first_stage(self, query: str, n: int) -> list[Hit]:
        mode = self.options.mode
        if mode == "dense":
            return self.store.search(self.embedder.embed_query(query), n)
        if mode == "sparse":
            return self.store.search_sparse(self.sparse.encode_query(query), n)
        dense, sparse = self.store.search_both(
            self.embedder.embed_query(query), self.sparse.encode_query(query), n
        )
        return rrf([dense, sparse], self.options.rrf_k)

    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        if self.reranker is None or not hits:
            return list(hits)
        cands = hits[: self.options.candidates]
        scores = self.reranker.scores(question, [h.chunk.text for h in cands])
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        return [Hit(cands[i].chunk, scores[i]) for i in order]

    def retrieve(self, question: str, k: int) -> list[Hit]:
        query = expand_abbreviations(question) if self.options.expand_abbreviations else question
        hits = self.first_stage(query, self._n_children(k))
        hits = self.rerank(question, hits)  # the cross-encoder reads the question as asked
        if self.options.parent_mode:
            hits = to_parents(hits)
        return cap_per_book(hits, self.options.max_per_book, k)

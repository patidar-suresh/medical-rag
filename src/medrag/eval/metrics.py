"""Ranking metrics for retrieval evaluation (binary relevance)."""

import math
from collections.abc import Sequence


def first_relevant_rank(relevant: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant result, or None if none is relevant."""
    return next((i for i, r in enumerate(relevant, 1) if r), None)


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    """Share of queries with at least one relevant result in the top k (a.k.a. Recall@k for one gold)."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None], k: int) -> float:
    """Mean reciprocal rank of the first relevant result, counting only the top k."""
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def ndcg_at_k(relevant: Sequence[bool], n_relevant_total: int, k: int) -> float:
    """nDCG@k for one query. `n_relevant_total` = relevant chunks in the whole corpus (>= 1)."""
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(relevant[:k], 1) if r)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(n_relevant_total, k) + 1))
    return dcg / ideal if ideal else 0.0

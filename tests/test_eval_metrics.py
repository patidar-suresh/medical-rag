import math

import pytest

from medrag.eval.common import normalize, percentile, token_recall, wilson_interval
from medrag.eval.metrics import first_relevant_rank, hit_rate, mrr, ndcg_at_k
from medrag.eval.nli import AnswerGrounding, SentenceGrounding, citations, split_sentences, strip_citations


def test_ranks_and_hit_rate() -> None:
    assert first_relevant_rank([False, True, True]) == 2
    assert first_relevant_rank([False, False]) is None
    ranks = [1, 3, None, 6]
    assert hit_rate(ranks, 1) == 0.25
    assert hit_rate(ranks, 5) == 0.5
    assert hit_rate(ranks, 10) == 0.75


def test_mrr_counts_only_top_k() -> None:
    assert mrr([1, 2, None], 10) == pytest.approx((1 + 0.5) / 3)
    assert mrr([1, 20], 10) == pytest.approx(0.5)


def test_ndcg() -> None:
    assert ndcg_at_k([True, False, False], 1, 10) == 1.0
    assert ndcg_at_k([False, True], 1, 10) == pytest.approx(1 / math.log2(3))
    # two relevant in corpus, only one retrieved at rank 1 -> dcg 1, idcg 1 + 1/log2(3)
    assert ndcg_at_k([True, False], 2, 10) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([False] * 10 + [True], 1, 10) == 0.0


def test_percentile_and_wilson() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 50) is None
    lo, hi = wilson_interval(50, 100) or (0, 0)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)
    assert wilson_interval(0, 0) is None


def test_normalize_matches_across_punctuation_and_case() -> None:
    assert normalize("Type II pneumocytes") in normalize("…by alveolar type-II  Pneumocytes, which…")


def test_token_recall() -> None:
    assert token_recall("HMG-CoA reductase", "The enzyme is HMG-CoA reductase [1].") == 1.0
    assert token_recall("axillary nerve", "It is the radial nerve.") == 0.5


def test_citation_parsing() -> None:
    s = "Ethosuximide is first line [1][2, 3]. See [note]."
    assert citations(s) == [1, 2, 3]
    assert strip_citations("Ethosuximide is first line [1].") == "Ethosuximide is first line."
    assert split_sentences("Short. This is one sentence [1]. And another one here.") == [
        "This is one sentence [1].",
        "And another one here.",
    ]


def test_grounding_aggregates() -> None:
    g = AnswerGrounding(
        [
            SentenceGrounding("a b c [1].", [0.9, 0.1], [1]),  # supported, correct cite
            SentenceGrounding("d e f [2].", [0.8, 0.2], [2]),  # supported, wrong cite
            SentenceGrounding("g h i [7].", [0.1, 0.1], [7]),  # unsupported, invalid cite
        ],
        0.5,
    )
    assert g.faithfulness == pytest.approx(2 / 3)
    assert g.citation_counts() == (3, 3, 2, 1)


def test_mcnemar_exact() -> None:
    from medrag.eval.common import mcnemar_p

    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert abs(mcnemar_p(10, 0) - 2 / 2**10) < 1e-12
    assert mcnemar_p(67, 21) < 0.0001  # Phase 3 parent-child vs baseline

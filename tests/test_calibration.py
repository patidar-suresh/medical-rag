import json
from pathlib import Path

import pytest

from medrag.eval.calibration import best_threshold, confusion, lexical_support
from medrag.eval.labeling import (
    Candidate,
    best_excerpt,
    candidates_from_run,
    label_loop,
    load_labels,
    stratified_sample,
)


def test_confusion_metrics() -> None:
    c = confusion([0.9, 0.8, 0.2, 0.1, 0.7], [True, False, True, False, True], 0.5)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 1, 1)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(2 / 3)
    assert c.balanced_accuracy == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_best_threshold_finds_separating_point() -> None:
    scores = [0.05, 0.1, 0.15, 0.3, 0.35, 0.9]
    labels = [False, False, False, True, True, True]
    t, c = best_threshold(scores, labels)
    assert 0.15 < t <= 0.3 and c.balanced_accuracy == 1.0


def test_lexical_support() -> None:
    assert (
        lexical_support("Axillary nerve supplies deltoid", ["The deltoid is supplied by the axillary nerve."])
        == 0.75
    )
    assert lexical_support("", ["x"]) == 0.0


def cand(key: str, score: float) -> Candidate:
    return Candidate(key, "q?", f"sentence {key}", ["A#0"], score)


def test_stratified_sample_balances_and_skips_done() -> None:
    cands = [cand(f"h{i}", 0.9) for i in range(10)] + [cand(f"l{i}", 0.1) for i in range(3)]
    picked = stratified_sample(cands, 6, done={"l0"})
    assert sum(c.nli_score >= 0.5 for c in picked) == 4  # only 2 low left, top up from high
    assert "l0" not in {c.key for c in picked}


def test_candidates_from_run(tmp_path: Path) -> None:
    rec = {
        "question": "q?",
        "sources": ["A#0", "B#1"],
        "sentences": [{"text": "Ethosuximide is first line [1].", "max_support": 0.8, "cited": [1]}],
    }
    (tmp_path / "generation.jsonl").write_text(json.dumps(rec) + "\n")
    [c] = candidates_from_run(tmp_path)
    assert c.sentence == "Ethosuximide is first line." and c.nli_score == 0.8 and len(c.key) == 16


def test_best_excerpt_prefers_overlapping_window() -> None:
    passage = (
        " ".join(["filler"] * 300) + " the deltoid is supplied by the axillary nerve " + " ".join(["x"] * 50)
    )
    overlap, excerpt = best_excerpt("The axillary nerve supplies the deltoid.", passage)
    assert overlap > 0.5 and "axillary" in excerpt


def test_label_loop_saves_each_answer_and_handles_keys(tmp_path: Path) -> None:
    out = tmp_path / "labels.jsonl"
    answers = iter(["?", "m", "y", "s", "n", "q"])  # help, more, yes, skip, no, quit
    shown: list[str] = []
    items = [cand("a", 0.9), cand("b", 0.2), cand("c", 0.3), cand("d", 0.8)]
    saved = label_loop(items, lambda cid: "passage text", out, lambda _: next(answers), shown.append)
    labels = load_labels(out)
    assert saved == 2 and [(lb.key, lb.supported) for lb in labels] == [("a", True), ("c", False)]
    assert any("passage [1]" in s for s in shown)  # 'm' printed full passages

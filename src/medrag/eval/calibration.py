"""Compare automatic support scores with human labels and pick a threshold.

Scorers compared on the same labelled sentences:
- `nli`: max entailment probability over the retrieved passages (what the eval uses);
- `lexical`: share of the sentence's content words found in the best passage (a cheap baseline);
- `nli_or_lexical`: supported if either says so (lexical at a fixed 0.9).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from medrag.eval.common import content_tokens


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Mean of recall on supported and on unsupported sentences: fair when classes are unequal."""
        tnr = self.tn / (self.tn + self.fp) if self.tn + self.fp else 0.0
        return (self.recall + tnr) / 2


def confusion(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Confusion:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels, strict=True):
        pred = s >= threshold
        tp += pred and y
        fp += pred and not y
        tn += not pred and not y
        fn += not pred and y
    return Confusion(tp, fp, tn, fn)


def best_threshold(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, Confusion]:
    """Threshold in 0.05..0.95 maximising balanced accuracy (ties -> closest to 0.5)."""
    grid = [round(0.05 * i, 2) for i in range(1, 20)]
    return max(
        ((t, confusion(scores, labels, t)) for t in grid),
        key=lambda tc: (round(tc[1].balanced_accuracy, 6), -abs(tc[0] - 0.5)),
    )


def lexical_support(sentence: str, passages: Sequence[str]) -> float:
    toks = content_tokens(sentence)
    if not toks:
        return 0.0
    best = 0.0
    for p in passages:
        vocab = set(content_tokens(p))
        best = max(best, sum(t in vocab for t in toks) / len(toks))
    return best

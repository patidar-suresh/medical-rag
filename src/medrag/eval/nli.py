"""Local groundedness checks with an open NLI (natural language inference) model.

For each answer sentence we ask: does a retrieved passage *entail* it? That gives
- faithfulness: share of answer sentences supported by any retrieved passage;
- citation precision: share of citations [n] whose passage n supports the sentence.

NLI models are trained on short premises, so long passages are split into
overlapping word windows and the best window counts.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

# A citation is a bracket holding only numbers and separators: [1], [2, 3], [1-2].
CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip())]
    return [p for p in parts if len(p.split()) >= min_words]


def citations(sentence: str) -> list[int]:
    nums: list[int] = []
    for m in CITATION_RE.finditer(sentence):
        nums.extend(int(n) for n in re.findall(r"\d+", m.group()))
    return nums


def strip_citations(sentence: str) -> str:
    text = re.sub(r"\s+", " ", CITATION_RE.sub("", sentence)).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def windows(text: str, size: int = 120, stride: int = 60) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - size + stride, stride)]


@dataclass(frozen=True)
class SentenceGrounding:
    sentence: str
    support: list[float]  # entailment probability per passage
    cited: list[int]  # 1-based passage numbers as written


@dataclass(frozen=True)
class AnswerGrounding:
    sentences: list[SentenceGrounding]
    threshold: float

    @property
    def faithfulness(self) -> float | None:
        if not self.sentences:
            return None
        return sum(max(s.support, default=0) >= self.threshold for s in self.sentences) / len(self.sentences)

    def citation_counts(self) -> tuple[int, int, int, int]:
        """(sentences with a citation, citations total, citations in range, citations supporting)."""
        with_cite = total = valid = supporting = 0
        for s in self.sentences:
            with_cite += bool(s.cited)
            for c in s.cited:
                total += 1
                if 1 <= c <= len(s.support):
                    valid += 1
                    supporting += s.support[c - 1] >= self.threshold
        return with_cite, total, valid, supporting


class NLIScorer:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        from medrag.hf import load_cached_first, quiet_hf

        quiet_hf()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._model = load_cached_first(
            lambda local: CrossEncoder(model_name, device=device, local_files_only=local)
        )
        if device.startswith("cuda"):
            self._model.model.half()
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        self._entail_idx = next(i for i, lbl in id2label.items() if lbl.startswith("entail"))
        self._batch_size = batch_size

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        probs = self._model.predict(
            list(pairs), batch_size=self._batch_size, apply_softmax=True, convert_to_numpy=True
        )
        return [float(p[self._entail_idx]) for p in probs]

    def support_matrix(self, hypotheses: Sequence[str], passages: Sequence[str]) -> list[list[float]]:
        """support[i][j] = best entailment of hypothesis i by any window of passage j."""
        passage_windows = [windows(p) for p in passages]
        pairs: list[tuple[str, str]] = []
        index: list[tuple[int, int]] = []  # (hypothesis i, passage j) per pair
        for i, hyp in enumerate(hypotheses):
            for j, wins in enumerate(passage_windows):
                for w in wins:
                    pairs.append((w, hyp))
                    index.append((i, j))
        support = [[0.0] * len(passages) for _ in hypotheses]
        for (i, j), sc in zip(index, self.entailment(pairs), strict=True):
            support[i][j] = max(support[i][j], sc)
        return support

    def ground(self, answer: str, passages: Sequence[str], threshold: float = 0.5) -> AnswerGrounding:
        sentences = split_sentences(answer)
        support = self.support_matrix([strip_citations(s) for s in sentences], passages)
        return AnswerGrounding(
            [SentenceGrounding(s, support[i], citations(s)) for i, s in enumerate(sentences)], threshold
        )

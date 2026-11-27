"""BM25 keyword vectors for Qdrant sparse search.

Qdrant stores one sparse vector per chunk and applies IDF itself (`Modifier.IDF`), computed
from the live collection, so only the term-frequency half of BM25 is computed here:

    doc weight(t)   = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl))
    query weight(t) = 1                                  (score = sum of idf(t) * doc weight)

Tokens are lowercased, stopword-filtered and Snowball-stemmed ("seizures" -> "seizur"), then
hashed to 32-bit ids, so no vocabulary file is needed. Collisions are rare and harmless.
"""

import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import Stemmer

# Runs of letters/digits. Hyphenated terms split ("beta-blocker" -> beta, blocker), so the
# hyphenated and spaced spellings match each other.
_TOKEN = re.compile(r"[a-z0-9]+")

# Short English stopword list. Negations ("no", "not") are dropped as well: BM25 can't use them.
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours yourself yourselves""".split()  # noqa: SIM905
)


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 100.0) -> None:
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl
        self._stemmer = Stemmer.Stemmer("english")

    def tokens(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        words = [w for w in _TOKEN.findall(text) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]
        return list(self._stemmer.stemWords(words))

    @staticmethod
    def _id(token: str) -> int:
        return zlib.crc32(token.encode())

    def fit(self, texts: Sequence[str]) -> "BM25Encoder":
        """Set the average document length (in tokens) used for length normalisation."""
        lengths = [len(self.tokens(t)) for t in texts]
        self.avgdl = max(1.0, sum(lengths) / max(1, len(lengths)))
        return self

    def _vector(self, weights: dict[int, float]) -> SparseVector:
        ids = sorted(weights)
        return SparseVector(ids, [weights[i] for i in ids])

    def encode_document(self, text: str) -> SparseVector:
        toks = self.tokens(text)
        norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
        weights: dict[int, float] = {}
        for tok, tf in Counter(toks).items():
            i = self._id(tok)
            weights[i] = weights.get(i, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return self._vector(weights)

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self.encode_document(t) for t in texts]

    def encode_query(self, text: str) -> SparseVector:
        return self._vector({self._id(t): 1.0 for t in set(self.tokens(text))})

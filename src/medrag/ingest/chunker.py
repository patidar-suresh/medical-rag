"""Fixed-size token-window chunking (the Phase 1 baseline).

Chunks are measured in the *embedding model's* tokens so no chunk is silently
truncated at embedding time. Chunk text is sliced from the original string using
token character offsets, so it keeps the source's exact wording and spacing.
Smarter, structure-aware chunking comes in Phase 3.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Maps a batch of strings to, per string, the (start_char, end_char) of each token.
OffsetsFn = Callable[[Sequence[str]], list[list[tuple[int, int]]]]


@dataclass(frozen=True)
class Span:
    start: int  # char offset into the book text
    end: int
    n_tokens: int


def token_offsets(text: str, offsets_fn: OffsetsFn) -> list[tuple[int, int]]:
    """Token offsets for a whole book, encoded line by line (batched) and shifted to book positions."""
    lines = text.split("\n")
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1
    result: list[tuple[int, int]] = []
    for start, offs in zip(line_starts, offsets_fn(lines), strict=True):
        result.extend((start + a, start + b) for a, b in offs if b > a)
    return result


def chunk_spans(offsets: list[tuple[int, int]], size: int, overlap: int) -> list[Span]:
    """Slide a window of `size` tokens with `overlap` tokens shared between neighbours."""
    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be smaller than chunk_size ({size})")
    step = size - overlap
    spans = []
    for i in range(0, len(offsets), step):
        window = offsets[i : i + size]
        spans.append(Span(window[0][0], window[-1][1], len(window)))
        if i + size >= len(offsets):
            break
    return spans


def hf_offsets_fn(model_name: str) -> OffsetsFn:
    """Offsets function backed by the embedding model's Hugging Face fast tokenizer."""
    from transformers import AutoTokenizer

    from medrag.hf import load_cached_first, quiet_hf

    quiet_hf()
    tok = load_cached_first(lambda local: AutoTokenizer.from_pretrained(model_name, local_files_only=local))
    tok.model_max_length = 10**9  # we tokenize whole lines here, not model inputs; avoid length warnings

    def fn(texts: Sequence[str]) -> list[list[tuple[int, int]]]:
        enc = tok(list(texts), add_special_tokens=False, return_offsets_mapping=True)
        return [[(int(a), int(b)) for a, b in offs] for offs in enc["offset_mapping"]]

    return fn

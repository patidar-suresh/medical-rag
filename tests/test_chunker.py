import pytest
from fakes import whitespace_offsets

from medrag.ingest.chunker import chunk_spans, token_offsets


def test_token_offsets_map_back_to_original_text() -> None:
    text = "alpha beta\n\ngamma  delta\nepsilon"
    offs = token_offsets(text, whitespace_offsets)
    assert [text[a:b] for a, b in offs] == ["alpha", "beta", "gamma", "delta", "epsilon"]


def test_windows_have_size_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(10))
    offs = token_offsets(text, whitespace_offsets)
    spans = chunk_spans(offs, size=4, overlap=1)
    chunks = [text[s.start : s.end].split() for s in spans]
    assert chunks == [["w0", "w1", "w2", "w3"], ["w3", "w4", "w5", "w6"], ["w6", "w7", "w8", "w9"]]
    assert all(s.n_tokens == 4 for s in spans)


def test_last_partial_window_kept_and_no_empty_tail() -> None:
    offs = token_offsets(" ".join(f"w{i}" for i in range(7)), whitespace_offsets)
    spans = chunk_spans(offs, size=4, overlap=1)
    assert [s.n_tokens for s in spans] == [4, 4]  # w0-w3, w3-w6: covers everything once


def test_text_shorter_than_one_chunk() -> None:
    spans = chunk_spans(token_offsets("just three words", whitespace_offsets), size=10, overlap=2)
    assert len(spans) == 1 and spans[0].n_tokens == 3


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_spans(token_offsets("", whitespace_offsets), size=10, overlap=2) == []


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_spans([(0, 1)], size=4, overlap=4)

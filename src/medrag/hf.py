"""Hugging Face model loading that stays offline once a model is cached.

The first use of a model downloads it (no account or token needed). After that we
load from the local cache only, so runs make no network calls.
"""

import os
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def quiet_hf() -> None:
    """Silence download/loading progress bars from huggingface_hub and transformers."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from transformers.utils import logging as tf_logging

    tf_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    tf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]


def load_cached_first(load: Callable[[bool], T]) -> T:
    """Call `load(local_files_only)`: try the local cache, download only if the model is missing."""
    try:
        return load(True)
    except OSError:  # not cached yet
        return load(False)

"""Environment health checks behind `medrag doctor`.

Each check returns a `CheckResult` instead of raising, so the CLI can report
every problem at once.
"""

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient

from medrag.config import Settings
from medrag.generate.llm import OllamaClient


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_qdrant(settings: Settings) -> CheckResult:
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=5, check_compatibility=False)
        info = client.info()
        n = len(client.get_collections().collections)
        return CheckResult("Qdrant", True, f"v{info.version} at {settings.qdrant_url}, {n} collection(s)")
    except Exception as e:  # noqa: BLE001 - report any failure as a failed check
        return CheckResult(
            "Qdrant",
            False,
            f"unreachable at {settings.qdrant_url} ({type(e).__name__}). "
            "Start it with: docker compose up -d qdrant",
        )


def check_ollama(settings: Settings, llm: OllamaClient | None = None) -> CheckResult:
    try:
        models = (llm or OllamaClient(settings)).installed_models()
    except Exception as e:  # noqa: BLE001
        return CheckResult("Ollama", False, f"unreachable at {settings.ollama_url} ({type(e).__name__})")
    if settings.llm_model not in models:
        return CheckResult(
            "Ollama",
            False,
            f"model '{settings.llm_model}' not pulled. Run: ollama pull {settings.llm_model} "
            f"(installed: {', '.join(models) or 'none'})",
        )
    return CheckResult("Ollama", True, f"model '{settings.llm_model}' available at {settings.ollama_url}")


def check_gpu() -> CheckResult:
    if not shutil.which("nvidia-smi"):
        return CheckResult("GPU", False, "nvidia-smi not found; embeddings will run on CPU", required=False)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return CheckResult("GPU", True, out, required=False)
    except (subprocess.SubprocessError, OSError) as e:
        return CheckResult("GPU", False, f"nvidia-smi failed ({type(e).__name__})", required=False)


def check_corpus(settings: Settings) -> CheckResult:
    files = sorted(settings.corpus_dir.glob("*.txt")) if settings.corpus_dir.is_dir() else []
    if not files:
        return CheckResult("Corpus", False, f"no .txt files in {settings.corpus_dir}")
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    return CheckResult("Corpus", True, f"{len(files)} books, {size_mb:.0f} MB in {settings.corpus_dir}")


def run_checks(settings: Settings) -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        lambda: check_qdrant(settings),
        lambda: check_ollama(settings),
        check_gpu,
        lambda: check_corpus(settings),
    ]
    return [c() for c in checks]

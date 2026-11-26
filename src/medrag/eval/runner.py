"""Run bookkeeping: each run gets eval/runs/<timestamp>-<model>/ with config, records, summary, report."""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from medrag.config import Settings
from medrag.eval.common import write_jsonl
from medrag.eval.suites import SuiteResult

RUNS_DIR = Path("eval/runs")

# (suite, metric, column header) shown by `medrag eval compare`.
KEY_METRICS: list[tuple[str, str, str]] = [
    ("retrieval", "hit@5", "ret hit@5"),
    ("retrieval", "mrr@10", "ret MRR"),
    ("retrieval", "ndcg@10", "ret nDCG"),
    ("generation", "faithfulness", "faithful"),
    ("generation", "citation_precision", "cite prec"),
    ("generation", "answer_recall", "ans recall"),
    ("generation", "false_refusal_rate", "false ref"),
    ("smoke", "pass_rate", "smoke"),
    ("smoke", "correct_refusal_rate", "corr ref"),
    ("medqa", "acc_closed_book", "MedQA cb"),
    ("medqa", "acc_rag", "MedQA rag"),
    ("generation", "ttft_p50_s", "TTFT s"),
    ("generation", "tokens_per_s", "tok/s"),
]


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


class Run:
    def __init__(self, settings: Settings, options: dict[str, Any], root: Path = RUNS_DIR) -> None:
        safe_model = re.sub(r"[^A-Za-z0-9.]+", "-", settings.llm_model)
        self.dir = root / f"{datetime.now():%Y%m%d-%H%M%S}-{safe_model}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.summary: dict[str, Any] = {}
        self.config = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "settings": settings.model_dump(mode="json"),
            "options": options,
            "git": _git_state(),
        }
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2))

    def add(self, suite: str, result: SuiteResult, seconds: float) -> None:
        result.summary["suite_seconds"] = round(seconds, 1)
        self.summary[suite] = result.summary
        write_jsonl(self.dir / f"{suite}.jsonl", result.records)
        self._write()

    def _write(self) -> None:
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=2))
        (self.dir / "report.md").write_text(render_report(self.config, self.summary))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    return "-" if v is None else str(v)


def render_report(config: dict[str, Any], summary: dict[str, Any]) -> str:
    s = config["settings"]
    lines = [
        f"# Eval run {config['started']}",
        "",
        f"- LLM: `{s['llm_model']}` (num_ctx {s['llm_num_ctx']}, temperature {s['llm_temperature']})",
        f"- Embeddings: `{s['embed_model']}`; chunks {s['chunk_size']}/{s['chunk_overlap']} tokens; "
        f"top_k {s['top_k']}",
        f"- Git: {config['git']['commit']}{' (uncommitted changes)' if config['git']['dirty'] else ''}",
        f"- Options: `{json.dumps(config['options'])}`",
        "",
    ]
    for suite, metrics in summary.items():
        lines += [f"## {suite}", "", "| metric | value |", "|---|---|"]
        for k, v in metrics.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kk} {_fmt(vv)}" for kk, vv in v.items())
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")
    lines += [
        "## Notes",
        "- Retrieval gold = an exact evidence quote; any chunk containing it counts as relevant.",
        "- Faithfulness/citations use an NLI model per answer sentence; sentences starting with a "
        "pronoun can be under-credited.",
        "- MedQA accuracy CIs are Wilson 95% intervals: differences inside overlapping CIs are noise.",
    ]
    return "\n".join(lines) + "\n"


def compare(root: Path = RUNS_DIR, last: int = 10) -> list[list[str]]:
    """Table rows (header first) of key metrics for the most recent runs."""
    runs = sorted(p for p in root.glob("*/summary.json"))[-last:]
    header = ["run", "model", *[h for _, _, h in KEY_METRICS]]
    rows = [header]
    for p in runs:
        summary = json.loads(p.read_text())
        cfg = json.loads((p.parent / "config.json").read_text())
        row = [p.parent.name[:15], cfg["settings"]["llm_model"]]
        row += [_fmt(summary.get(suite, {}).get(metric)) for suite, metric, _ in KEY_METRICS]
        rows.append(row)
    return rows

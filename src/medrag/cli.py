"""`medrag` command-line interface."""

import typer

from medrag.config import get_settings
from medrag.doctor import run_checks
from medrag.logs import configure_logging

app = typer.Typer(help="MedRAG: local medical RAG over textbooks.", no_args_is_help=True)


@app.callback()
def main() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


@app.command()
def doctor() -> None:
    """Check that Qdrant, Ollama (and the configured model), GPU and corpus are ready."""
    results = run_checks(get_settings())
    for r in results:
        mark = "OK  " if r.ok else ("FAIL" if r.required else "WARN")
        typer.echo(f"[{mark}] {r.name:<7} {r.detail}")
    if any(r.required and not r.ok for r in results):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

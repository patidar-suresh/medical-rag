"""`medrag` command-line interface."""

import time
from typing import TYPE_CHECKING, Annotated

import typer

from medrag.cli_eval import eval_app
from medrag.config import Settings, get_settings
from medrag.doctor import run_checks
from medrag.logs import configure_logging

if TYPE_CHECKING:
    from medrag.index.embedder import Embedder
    from medrag.rag import RAGPipeline

app = typer.Typer(help="MedRAG: local medical RAG over textbooks.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


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


@app.command()
def ingest(
    books: Annotated[
        list[str] | None, typer.Option("--book", "-b", help="Only ingest this book (repeatable).")
    ] = None,
) -> None:
    """Chunk, embed and index the textbooks into Qdrant (rebuilds the collection)."""
    from medrag.generate.llm import OllamaClient
    from medrag.index.embedder import SentenceTransformerEmbedder
    from medrag.index.store import QdrantStore
    from medrag.ingest.chunker import hf_offsets_fn
    from medrag.ingest.loader import load_books
    from medrag.ingest.pipeline import ingest as run_ingest

    s = get_settings()
    loaded = load_books(s.corpus_dir, books)
    typer.echo(f"Loaded {len(loaded)} book(s) from {s.corpus_dir}")

    # The LLM and the embedder share one small GPU: free the LLM's memory first.
    try:
        OllamaClient(s).unload()
    except Exception:  # noqa: BLE001 - Ollama being down shouldn't block ingestion
        typer.echo("Note: could not unload the Ollama model (is Ollama running?)")

    embedder = SentenceTransformerEmbedder(s)
    typer.echo(f"Embedding with {s.embed_model} on {embedder.device}")
    stats = run_ingest(
        loaded,
        settings=s,
        offsets_fn=hf_offsets_fn(s.embed_model),
        embedder=embedder,
        store=QdrantStore.from_url(s.qdrant_url, s.collection),
        show_progress=True,
    )
    typer.echo(
        f"Indexed {stats.chunks} chunks ({stats.tokens:,} tokens) from {stats.books} book(s) "
        f"into '{s.collection}'. Time: chunking {stats.seconds_chunking:.0f}s, "
        f"embedding {stats.seconds_embedding:.0f}s, indexing {stats.seconds_indexing:.0f}s"
    )


def build_pipeline(s: Settings, embedder: "Embedder | None" = None) -> "RAGPipeline":
    """Pipeline from settings. Pass `embedder` to reuse an already-loaded model."""
    # Heavy imports (torch) stay inside commands so `medrag doctor` starts instantly.
    from medrag.generate.llm import OllamaClient
    from medrag.index.embedder import SentenceTransformerEmbedder
    from medrag.index.store import QdrantStore
    from medrag.rag import RAGPipeline

    store = QdrantStore.from_url(s.qdrant_url, s.collection)
    if not store.exists():
        typer.echo(f"Collection '{s.collection}' not found in Qdrant. Run `medrag ingest` first.", err=True)
        raise typer.Exit(code=1)
    return RAGPipeline(
        embedder or SentenceTransformerEmbedder(s), store, OllamaClient(s), s.top_k, s.parent_chunk_size > 0
    )


@app.command()
def ask(
    question: str,
    k: Annotated[int | None, typer.Option("--k", help="Passages to retrieve.")] = None,
    show_context: Annotated[bool, typer.Option(help="Print the retrieved passages.")] = False,
) -> None:
    """Answer a question from the indexed textbooks (streams the answer)."""
    rag = build_pipeline(get_settings())
    t0 = time.perf_counter()
    hits, tokens = rag.stream_answer(question, k)
    for i, h in enumerate(hits, 1):
        if show_context:
            typer.echo(f"--- [{i}] {h.chunk.chunk_id} (score {h.score:.3f})\n{h.chunk.text.strip()}\n")
    for t in tokens:
        typer.echo(t, nl=False)
    typer.echo("\n\nSources:")
    for i, h in enumerate(hits, 1):
        typer.echo(f"  [{i}] {h.chunk.chunk_id}  (score {h.score:.3f})")
    typer.echo(f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    app()

"""Load the textbook corpus from `corpus_dir/*.txt`."""

from dataclasses import dataclass
from pathlib import Path

# File-name prefix -> subject. Keys keep the corpus's own spellings.
SUBJECTS: dict[str, str] = {
    "Anatomy": "anatomy",
    "Biochemistry": "biochemistry",
    "Cell": "cell biology",
    "First": "usmle review",
    "Histology": "histology",
    "Immunology": "immunology",
    "InternalMed": "internal medicine",
    "Neurology": "neurology",
    "Obstentrics": "obstetrics",
    "Pathology": "pathology",
    "Pathoma": "pathology",
    "Pediatrics": "pediatrics",
    "Pharmacology": "pharmacology",
    "Physiology": "physiology",
    "Psichiatry": "psychiatry",
    "Surgery": "surgery",
}


@dataclass(frozen=True)
class Book:
    name: str  # file stem, e.g. "Pharmacology_Katzung"
    subject: str
    text: str


def subject_for(book_name: str) -> str:
    return SUBJECTS.get(book_name.split("_", 1)[0], "general")


def load_books(corpus_dir: Path, only: list[str] | None = None) -> list[Book]:
    """Load every `.txt` book, optionally restricted to names in `only` (case-insensitive)."""
    wanted = {n.lower() for n in only} if only else None
    books = []
    for path in sorted(corpus_dir.glob("*.txt")):
        if wanted is not None and path.stem.lower() not in wanted:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        books.append(Book(name=path.stem, subject=subject_for(path.stem), text=text))
    if wanted is not None:
        missing = wanted - {b.name.lower() for b in books}
        if missing:
            raise FileNotFoundError(f"Books not found in {corpus_dir}: {', '.join(sorted(missing))}")
    return books

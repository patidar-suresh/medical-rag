# Phase 3: cleaning and chunking experiments

**Decision:** clean the text, chunk by structure, embed **160-token children** and return their
**800-token parent** section to the LLM (no contextual headers). Retrieval hit@5 goes from
**0.59 → 0.74** (paired test p < 0.0001). Reproduce with `scripts/phase3_sweep.sh`.

## Setup

- Retrieval suite: 300 synthetic questions (`data/eval/synthetic.jsonl`), 11–23 per book.
  Gold = an exact evidence quote; a returned passage is relevant if it contains the quote
  (letters/digits only, after Unicode folding, so cleaning doesn't break matching).
- Embeddings `BAAI/bge-small-en-v1.5` (fp16), Qdrant cosine, top-k as stated.
- "Returned passage" is what the LLM sees: the parent in parent-child mode. Bigger passages make
  hits easier, so **context tokens@5** is reported next to every hit rate.
- Significance: exact McNemar test on per-question hit@5 against the baseline (questions that
  flipped hit→miss vs miss→hit). With 300 questions, ±0.03 differences are usually noise.

## Results

| Collection | Cleaning | Chunks | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 | ctx tok@5 | vs baseline @5 (gained/lost, p) |
|---|---|---|---|---|---|---|---|---|---|---|
| `medrag_chunks` (Phase 1) | – | fixed 500/50 | 0.313 | 0.483 | 0.590 | 0.697 | 0.429 | 0.471 | 2500 | – |
| `p3_fixed500_clean` | ✓ | fixed 500/50 | 0.287 | 0.510 | 0.560 | 0.657 | 0.410 | 0.457 | 2500 | 30/39, p=0.34 |
| `p3_struct500` | ✓ | structured 500 | 0.321 | 0.465 | 0.569 | 0.642 | 0.418 | 0.466 | 2033 | 35/42, p=0.49 |
| `p3_struct500_ctx` | ✓ | structured 500 + headers | 0.291 | 0.455 | 0.539 | 0.646 | 0.396 | 0.449 | 1830 | 36/52, p=0.11 |
| `p3_struct256` | ✓ | structured 256 | 0.337 | 0.537 | 0.650 | 0.740 | 0.460 | 0.526 | 1041 | – |
| `p3_struct256_ctx` | ✓ | structured 256 + headers | 0.330 | 0.547 | 0.640 | 0.737 | 0.462 | 0.523 | 868 | 55/40, p=0.15 |
| `p3_pc160_800_ctx` | ✓ | parent-child 160→800 + headers | 0.463 | 0.627 | 0.717 | 0.820 | 0.568 | 0.627 | 3734 | 66/28, p=0.0001 |
| **`p3_pc160_800`** | ✓ | **parent-child 160→800** | 0.445 | **0.659** | **0.743** | 0.796 | 0.566 | 0.620 | 3710 | **67/21, p<0.0001** |

Header ablations (same chunking with vs. without headers): 256 → 28 gained / 31 lost (p=0.79);
parent-child → 23 gained / 31 lost (p=0.34). **Headers make no measurable difference.**

hit@5 by subject (selected):

| Collection | USMLE review (First Aid) | Surgery | Pathology | Pharmacology | Anatomy |
|---|---|---|---|---|---|
| `medrag_chunks` | 0.34 | 0.60 | 0.52 | 0.55 | 0.57 |
| `p3_pc160_800_ctx` | 0.41 | 0.87 | 0.74 | 0.75 | 0.74 |
| `p3_pc160_800` | **0.56** | 0.80 | 0.67 | **0.85** | **0.87** |

## Findings

1. **Parent-child is the one change that clearly works.** Small children make the query–chunk
   match precise; the parent restores the surrounding context. It also wins at *equal context*:
   its top 3 parents (~2,200 tokens) hit 0.659 vs. the baseline's top 5 (2,500 tokens) at 0.590.
2. **Smaller chunks help on their own** (500 → 256: 0.57 → 0.65, using less than half the
   context), consistent with bge-small embedding short passages more faithfully.
3. **Cleaning alone doesn't move retrieval** (0.59 → 0.56, p=0.34; noise). It is still kept:
   it fixes real defects (ligatures, 8k broken words in DSM-5, 2k layout footers in Surgery)
   that matter for keyword search in Phase 4, and structured chunking needs its paragraphs.
4. **Contextual headers don't pay off here.** Breadcrumbs are noisy where heading detection is
   weak (First Aid: 0.56 without vs. 0.41 with), and a ~20-token prefix is a large share of a
   160-token child for a small embedding model. They also cost 45% more vectors (228k vs 158k).
   The breadcrumb is still stored and shown to the LLM as the passage's source label.

## Caveats

- **Home advantage for the baseline:** the synthetic questions were written from the baseline's
  own 500-token chunks, so any re-chunking starts at a disadvantage. That makes the parent-child
  win more convincing, but it may understate the other strategies.
- Questions written by a 3B model are sometimes vaguer than their evidence; absolute hit rates are
  less meaningful than differences between configurations on the same questions.
- Parent-child sends ~1.5x more context to the LLM (5 × 800 tokens). Its effect on answer
  quality and latency is in the downstream evaluation below.

## Downstream (full eval on the default index)

`llama3.2:3b`, top_k 5. Phase 1 = `eval/runs/20261126-164644-*`, Phase 3 = `eval/runs/20261126-223008-*`.

| Metric | Phase 1 | Phase 3 | Note |
|---|---|---|---|
| Faithfulness (NLI) | 0.31 | **0.51** | same uncalibrated checker on both, so the direction is meaningful |
| Citation precision | 0.20 | **0.42** | |
| Citation validity | 1.00 | 0.88 | model copies the books' figure refs ("Fig. 8-2" → "[8-2]"), see below |
| Answer recall | 0.67 | **0.73** | |
| Smoke pass / correct refusals | 0.92 / 1.00 | 0.92 / 1.00 | |
| MedQA closed-book / RAG | 0.58 / 0.65 | 0.57 / 0.61 | RAG paired: 7 better, 11 worse, p=0.48 (noise) |
| Prompt tokens (mean) | 2,646 | 3,920 | |
| Generation p50 / TTFT p50 | 5.7 s / 2.0 s | **10.2 s / 3.3 s** | |
| Decode speed | 17.0 tok/s | **10.6 tok/s** | longer context slows the 3B model on the 4 GB GPU |

**Trade-off:** answers are better grounded, at roughly twice the generation time. Options:
`MEDRAG_TOP_K=3` returns ~2,400 tokens (less than Phase 1's 2,500) and retrieval at k=3 (0.659)
still beats Phase 1 at k=5 (0.590); its effect on answers has not been measured yet.

**Citation collision (for Phase 6):** a third of parents contain figure references like
"(Fig. 8-2)"; the model imitates them in its `[n]` citation slots. Use a citation format the
books cannot produce (e.g. `[P1]`), or strip figure references in cleaning.

## Cost

| Collection | Vectors | Embed time | Index time |
|---|---|---|---|
| `medrag_chunks` | 43,633 | 4.2 min | 25 s |
| `p3_struct256` | 97,268 | 4.5 min | 56 s |
| `p3_pc160_800` | 157,752 | 4.6 min | 1.3–2.0 min |
| `p3_pc160_800_ctx` | 228,403 | 6.8 min | 2.5 min |

Payload size grows with parent-child, because each child stores its parent's text.
Qdrant keeps payload on disk (`on_disk_payload=True`), so RAM holds only the vectors
(158k × 384 floats ≈ 240 MB).

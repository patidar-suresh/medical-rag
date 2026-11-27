# Phase 4: hybrid retrieval and reranking

**Decision:** search both the dense index and a BM25 keyword index, fuse them with Reciprocal
Rank Fusion, then rerank the top 50 children with the `cross-encoder/ms-marco-MiniLM-L-6-v2`
cross-encoder. Embeddings stay `bge-small`. Retrieval hit@5 goes **0.74 → 0.87** and hit@1
**0.44 → 0.63** against Phase 3 (paired test p < 0.0001), at **+50 ms** per query.
Reproduce with `scripts/phase4_sweep.sh`, `scripts/phase4_embeddings.sh` and `scripts/phase4_medqa.sh`.

## Setup

- Index: the Phase 3 default (structured parent-child, 160-token children → 800-token parents),
  re-ingested with a BM25 sparse vector next to each dense vector. Dense-only search on the new
  index reproduces Phase 3 exactly (hit@5 0.743), so every difference below comes from retrieval.
- BM25: Snowball-stemmed, stopword-filtered tokens hashed to ids; we store the BM25 term-frequency
  weights and Qdrant applies IDF from the live collection (`Modifier.IDF`). No extra service.
- Hybrid: dense top 50 and BM25 top 50 in one batched Qdrant call, fused client-side with RRF
  (k = 60). Reranking reads the 160-token children; the parents are what the LLM sees.
- Same 300 synthetic questions and gold-quote matching as Phase 3. Significance: exact McNemar
  test on paired per-question outcomes, now built in: `medrag eval paired <run A> <run B>`.
- Latency is per query on an RTX A500 (4 GB), embedding + search + rerank, warm models.

## Retrieval results

| Pipeline | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| Phase 3: dense | 0.441 | 0.656 | 0.743 | 0.796 | 0.564 | 0.618 | 37 ms |
| BM25 only | 0.582 | 0.743 | 0.809 | 0.880 | 0.679 | 0.725 | 10 ms |
| Hybrid (RRF) | 0.505 | 0.719 | 0.803 | 0.886 | 0.632 | 0.692 | 40 ms |
| Dense + MiniLM rerank | 0.612 | 0.786 | 0.829 | 0.893 | 0.708 | 0.751 | 71 ms |
| BM25 + MiniLM rerank | 0.629 | 0.799 | 0.860 | 0.920 | 0.728 | 0.772 | 63 ms |
| **Hybrid + MiniLM rerank** | **0.625** | **0.806** | **0.866** | **0.930** | **0.727** | **0.774** | **87 ms** |
| Hybrid + MedCPT cross-encoder | 0.518 | 0.753 | 0.840 | 0.920 | 0.655 | 0.717 | 239 ms |
| Hybrid + bge-reranker-base | 0.448 | 0.669 | 0.766 | 0.876 | 0.583 | 0.653 | 279 ms |
| Hybrid + bge-reranker-v2-m3 | 0.682 | 0.856 | 0.906 | 0.940 | 0.776 | 0.816 | 854 ms |

Paired hit@5 comparisons (gained / lost questions):

| A → B | gained / lost | p |
|---|---|---|
| Phase 3 dense → hybrid + MiniLM | 47 / 10 | < 0.0001 |
| Phase 3 dense → BM25 only | 45 / 25 | 0.02 |
| Hybrid → hybrid + MiniLM | 29 / 10 | 0.003 |
| BM25 + MiniLM → hybrid + MiniLM | 5 / 3 | 0.73 (hit@10: 7 / 4, p = 0.55) |
| Hybrid + MiniLM → hybrid + bge-reranker-v2-m3 | 16 / 4 | 0.01 |

Variations on hybrid + MiniLM, none different from it (all p ≥ 0.2):

| Variation | hit@5 | hit@10 | p50 latency |
|---|---|---|---|
| 25 rerank candidates | 0.863 | 0.926 | 68 ms |
| 100 rerank candidates | 0.870 | 0.936 | 175 ms |
| + abbreviation expansion | 0.863 | 0.926 | 92 ms |
| + per-book cap of 2 | 0.866 | 0.900 | 90 ms |
| bge-base embeddings instead of bge-small | 0.866 | 0.943 | – |

## Embedding bake-off (dense only, identical chunks)

Chunk boundaries were measured with the bge-small tokenizer for every model
(`MEDRAG_CHUNK_TOKENIZER`), so only the embedding model changes.

| Model | Params | Dim | hit@1 | hit@5 | hit@10 | Embed time (158k children) |
|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 33M | 384 | 0.441 | 0.743 | 0.796 | 7.8 min |
| `BAAI/bge-base-en-v1.5` | 110M | 768 | 0.495 | 0.769 | 0.866 | 12.0 min |
| `ncbi/MedCPT` (article + query encoders) | 2 × 110M | 768 | 0.258 | 0.488 | 0.639 | 15.2 min |

- bge-base is better on its own (hit@1 p = 0.03, hit@10 p = 0.0005), but **inside the full
  pipeline the gain disappears** (hit@5 0.866 vs 0.866): BM25 and the reranker recover what the
  smaller model misses. bge-small stays, at a third of the indexing cost and VRAM.
- MedCPT does poorly here. It was trained on PubMed search logs (short queries → abstracts)
  and scores with an unnormalised dot product; textbook passages and our cosine index are both
  off its training distribution. Not pursued further.
- `BAAI/bge-m3` was skipped: 568M parameters (17× bge-small), only a pickle checkpoint, ~1 h to
  index on this GPU and too large to sit next to the LLM. Worth a re-test on a bigger GPU.

## Findings

1. **The reranker is the biggest single win.** A 22M-parameter cross-encoder reading question
   and passage together adds +0.06 hit@5 and +0.12 hit@1 over hybrid search, for ~50 ms.
2. **Keyword search matters for this corpus.** BM25 alone beats dense embeddings (0.81 vs 0.74).
   Medical text is full of exact terms (drug names, eponyms, lab values) that a small embedding
   model blurs. *Caveat:* the synthetic questions were written from the passages, so they share
   their wording, which favours BM25; MedQA below is the less biased check.
3. **Hybrid's value is recall.** Before reranking, hybrid ≈ BM25 at hit@5 but has the best
   hit@10, i.e. a better candidate pool. After reranking, hybrid vs BM25-only isn't
   significant; hybrid is kept because dense search covers paraphrased questions that share no
   words with the text, which this test set under-represents.
4. **Bigger rerankers are not automatically better.** bge-reranker-base *hurts* (0.77 vs 0.80
   without reranking), with fp16 and score saturation ruled out. bge-reranker-v2-m3 is the most
   accurate (0.91) but needs ~2.6 GB of VRAM and 0.85 s per query: the right choice on an 8 GB+
   GPU (`MEDRAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`), not next to a 3B LLM on 4 GB.
5. **Diversity and abbreviation expansion: no effect on this test set.** A per-book cap can only
   help when one book crowds out a better one; with good reranking that is rare, and it lowers
   hit@10 (0.93 → 0.90). Both stay available, off by default. Abbreviations were also measured
   on MedQA, where vignettes use them heavily (see below).
6. **Not built: subject routing.** Classifying the question's subject to boost matching books
   addresses the same "wrong book" failure as the cap; with right-book@5 at 0.95 after
   reranking there is little left for it to fix.

## MedQA and full evaluation

_Pending: `scripts/phase4_medqa.sh` (300 questions per run)._

## Licences

| Component | Licence |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 weights; **trained on MS MARCO, whose terms are non-commercial**. For commercial use, check this or switch to the MedCPT cross-encoder (US government work, public domain; hit@5 0.84) |
| `BAAI/bge-reranker-v2-m3`, `bge-reranker-base` | Apache-2.0 / MIT |
| PyStemmer (Snowball) | MIT / BSD |

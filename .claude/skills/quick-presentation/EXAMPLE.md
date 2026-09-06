---
marp: true
theme: quick-presentation
paginate: true
title: "Retrieval-Augmented Generation: A Practical Overview"
author: "Data Science Team"
date: "2026-04-21"
---

<!-- _class: title -->

# Retrieval-Augmented Generation
## A Practical Overview

Data Science Team — 2026-04-21

---

## Agenda

1. What RAG is and why it matters
2. System architecture
3. Retrieval quality in practice
4. Evaluation snapshot
5. Takeaways

---

<!-- _class: section -->

# 1. What RAG is and why it matters

---

## Context

- LLMs trained on static corpora drift out of date within months.
- Fine-tuning every N weeks is expensive and fragile.
- Retrieval-Augmented Generation (RAG) injects fresh, domain-specific context at inference time.

*Key claim: RAG turns a frozen model into a dynamic system without retraining.*

---

## Architecture

![RAG system architecture with query encoder, vector store, and generator](assets/diagrams/1-rag-architecture.svg)

<!-- Source: assets/diagrams/1-rag-architecture.mmd -->

---

<!-- _class: section -->

# 2. Retrieval quality in practice

---

## Minimal retrieval loop

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

_encoder = SentenceTransformer("all-MiniLM-L6-v2")
_client = QdrantClient(url="http://localhost:6333")


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the top-k nearest chunks for the given query."""
    vector = _encoder.encode(query).tolist()
    hits = _client.search(collection_name="docs", query_vector=vector, limit=k)
    return [hit.payload for hit in hits]
```

---

<!-- Two-column layout using Marp's column directive -->

<div class="columns">

<div>

### Sparse retrievers

- BM25, SPLADE
- Deterministic, explainable
- Strong on rare tokens
- Weak on paraphrase

</div>

<div>

### Dense retrievers

- Bi-encoders, ColBERT
- Semantic similarity
- Robust to paraphrase
- Needs retraining per domain

</div>

</div>

*Hybrid retrieval (dense + sparse with RRF) is the current production default.*

---

<!-- _class: section -->

# 3. Evaluation snapshot

---

## Retrieval quality on internal eval set

![Recall@k chart comparing BM25, dense, and hybrid retrievers](assets/charts/1-recall-at-k.png)

- Hybrid beats dense by ~6 points Recall@10.
- Gap widens for low-frequency entity queries.

---

## Callout

> "Retrieval quality sets the ceiling for generation quality."
> — internal RAG post-mortem, Q1 2026

---

<!-- _class: section -->

# 4. Takeaways

---

## Takeaways

1. RAG is a system, not a model — treat it like one.
2. Measure retrieval **before** tuning generation.
3. Hybrid retrieval is the default; revisit yearly.
4. Keep diagram sources (`.mmd`) next to rendered SVGs for future edits.
5. Version control the deck like code.

---

## Next steps

- Prototype hybrid retrieval on our domain corpus.
- Add Recall@k + answer faithfulness to the eval dashboard.
- Revisit chunking strategy in the follow-up spike.

---

<!-- _class: title -->

# Thank you

Questions?

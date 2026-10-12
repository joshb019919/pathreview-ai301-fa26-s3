# Plan

## Environment

Ubuntu 26.04.1
Linux 7.0.0-34-generic
Ollama v0.35.1
Llama 3.1 8B

## Changes

I propose one new file, `rag/retriever/reranker.py`.  In an `LLMReranker` class,
`rerank(query, chunks, top_k=10)` takes an integer relevance score from the the
Llama model, (0 to 10) per chunk, sorts by it and returns the top `top_k`.  By
default, it's not used (`enabled=False`).  It returns `chunks[:top_k]` unchanged
to match the same returned by `HybridRetriever.retrieve()`.  Only `score` is
replaced by the model.  If the model's reply can't be parsed, the original order 
is returned.  `hybrid.py` and `review_generator.py` are not modified.

The caller has to get more than the default 10 chunks returned to have something
to re-rank (`max_chunks=10` in `hybrid.py:117`).  The option to return 
`[:top_k]` guarantees it'll give the generator what it needs.


## Tests

To test, I'll add `test_reranking.py` to `tests/unit` with a mocked LLM client, 
It'll check the chunk order, the `top_k` cap, dict shape within the list, that
`enabled=False` passes through just fine, what happens if it gets a bad reply, 
and what happens in the event of a tie. `rag/evaluator/` seems like it would be
a good tool for this, but it returns one keyword-overlap score for the whole 
chunk set and can't show a reorder.

## Unknowns

I couldn't find a Python caller that connects retriever and
generator, so I'm leaving wiring things up out of this change.

## Questions 
The issue says "smaller LLM", but these days that could be a lot of
sizes.  Is 8 billion parameters small enough for this use case?

## AI Use

I'm using Claude Code to read the repo and check this plan, and I
review everything it produces before posting.

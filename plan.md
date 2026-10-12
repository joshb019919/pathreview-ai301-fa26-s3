# Plan: LLM Re-Ranker for Retrieved Chunks (Issue #10)

## Environment

OS: Ubuntu 26.04.1
Linux 7.0.0-34-generic
Ollama v0.35.1
Llama 3.1 8B (`ollama pull llama3.1:8b`)
Unmodified `main` repo state

## Observed in the Repo

`HybridRetriever.retrieve()` (`rag/retriever/hybrid.py:34`) takes
`max_chunks: int = 10` and returns `results[:max_chunks]` (line 117).
`ReviewGenerator.generate_full_review(profile_data, retrieved_chunks)`
(`rag/generator/review_generator.py:87`) uses `chunks[:10]` for
context (line 148) and `retrieved_chunks[:5]` for citations (line 170).

`ReviewGenerator` talks to its LLM through `openai.OpenAI(api_key, base_url)`.
Ollama serves an OpenAI-compatible endpoint, so the reranker uses the same
client type.

I did not find a Python caller that connects retriever to generator.
`docs/ARCHITECTURE.md` describes the connection, but I'm not sure how it's
wired up.

## Scope

One new file, `rag/retriever/reranker.py`, plus tests.

`plan.md` (this file) and `comment.md` are also new files, both added to the
root directory of this repo.

Changes to `hybrid.py`, `review_generator.py`, the evaluator, fine-tuning, or 
wiring the reranker into the running app will not be required.

### Interface

```python
class LLMReranker:
    def __init__(self, client, model: str, enabled: bool = False): ...
    def rerank(self, query: str, chunks: list[dict], top_k: int = 10) -> list[dict]: ...
```

`enabled=False` (default): `rerank()` returns `chunks[:top_k]` unchanged, so
the existing behavior is untouched and the feature stays optional.

`enabled=True`: the LLM scores each chunk's relevance to the query as an
integer from 0 to 10. The output is sorted by that score (highest first,
ties keep the retriever's original order) and cut to `top_k`.

To rerank more than the default 10, the caller has to retrieve more first,
e.g. `retrieve(..., max_chunks=30)`, then `rerank(..., top_k=10)`.

Each returned chunk keeps the retriever's shape: `{id, text, metadata, score,` 
`vector_score, keyword_score}`.

Only `score` is replaced by the LLM score. `vector_score` and
`keyword_score` are left as they were.

The default `top_k=10` matches the generator's `[:10]` context slice.

### Prompt and parsing

One call to `client.chat.completions.create` with a system prompt that
describes the task and a user prompt listing the query and the numbered
chunks. The model is asked to return only a JSON list of integers, one per
chunk, in order.

If the reply is not valid JSON, has the wrong length, or has a value outside
0 to 10, the reranker logs a warning and returns `chunks[:top_k]` in the
original order. A bad LLM reply must never drop or reorder chunks by accident.
Prompt text lives as a constant at the top of `reranker.py`.

## Tests

New file `tests/unit/test_reranking.py`. The LLM client is mocked, so no
Ollama is needed and the `test-unit` CI job stays green.

Mock returns scores `[2, 9, 5]` for three chunks: output order is
chunk 2, chunk 3, chunk 1.

`top_k=2` returns exactly 2 chunks, the two highest-scoring.

`enabled=False` returns the input order unchanged and the mock is never
called.

Each returned dict has the keys `id, text, metadata, score, vector_score,
keyword_score`, and `vector_score`/`keyword_score` are unchanged.

Mock returns non-JSON, a list of the wrong length, or an out-of-range
value: original order is returned.

Tied scores keep the retriever's order.

To Run: `pytest tests/unit/test_reranking.py`, then `make check && make test-unit`
to confirm lint, types and the full unit suite.

Manual check against a real model (not part of CI): with Ollama running,
retrieve with `max_chunks=30`, call `rerank(..., top_k=10)` with
`enabled=True`, and print the chunk ids before and after. The order should
differ from the retriever's order.

I considered using `rag/evaluator/` (`RelevanceScorer`, `FaithfulnessChecker`)
for the pass/fail check and decided against it. Both return one float over the
whole chunk set from keyword overlap and have no thresholds, so they can't show
whether a reranker changed the order. I may report their before and after
numbers as extra information only.

### Evidence

`make check` (ruff, black, mypy). The venv's console scripts had a stale
shebang after the repo moved, so I ran the same Makefile targets with `VENV_BIN` pointed at
wrappers that call `.venv/bin/python -m <tool>`. Nothing in the repo was changed.

```
$ make check
.venv/bin/ruff check .
All checks passed!
.venv/bin/black .
All done! ✨ 🍰 ✨
112 files left unchanged.
.venv/bin/mypy api/ core/ ingestion/ rag/ agent/ safety/
Success: no issues found in 77 source files
```

`make test-unit` (per-test lines for the other test files omitted; all `test_reranking.py` lines kept):

```
$ make test-unit
.venv/bin/pytest tests/unit -v -m unit
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0 -- /home/user/Documents/github/codepath/pathreview-ai301-fa26-s3/.venv/bin/python
cachedir: .pytest_cache
benchmark: 5.3.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
hypothesis profile 'default'
rootdir: /home/user/Documents/github/codepath/pathreview-ai301-fa26-s3
configfile: pyproject.toml
plugins: pytest_httpserver-1.2.0, benchmark-5.3.0, hypothesis-6.168.5, asyncio-1.4.0, anyio-4.15.1, cov-7.1.0, platformdirs-4.12.3
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 444 items

tests/unit/test_batch_processor.py::TestBatchEmbeddingProcessor::test_empty_chunks_list_returns_empty XFAIL [  0%]
tests/unit/test_batch_processor.py::TestBatchEmbeddingProcessor::test_normal_chunks_list_processes PASSED [  0%]
... (remaining PASSED/XFAIL lines omitted) ...
tests/unit/test_reranking.py::TestLLMReranker::test_orders_by_llm_score_descending PASSED [ 67%]
tests/unit/test_reranking.py::TestLLMReranker::test_top_k_caps_to_highest_scoring PASSED [ 67%]
tests/unit/test_reranking.py::TestLLMReranker::test_disabled_passes_through_without_calling_llm PASSED [ 67%]
tests/unit/test_reranking.py::TestLLMReranker::test_disabled_still_applies_top_k PASSED [ 68%]
tests/unit/test_reranking.py::TestLLMReranker::test_dict_shape_and_other_scores_unchanged PASSED [ 68%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[not json] PASSED [ 68%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[[1, 2]] PASSED [ 68%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[[1, 2, 3, 4]] PASSED [ 68%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[[1, 2, 11]] PASSED [ 69%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[[1, 2, -1]] PASSED [ 69%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[[1, 2, 3.5]] PASSED [ 69%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[[1, 2, true]] PASSED [ 69%]
tests/unit/test_reranking.py::TestLLMReranker::test_bad_reply_returns_original_order[{"scores": [1, 2, 3]}] PASSED [ 70%]
tests/unit/test_reranking.py::TestLLMReranker::test_client_error_returns_original_order PASSED [ 70%]
tests/unit/test_reranking.py::TestLLMReranker::test_ties_keep_retriever_order PASSED [ 70%]
tests/unit/test_reranking.py::TestLLMReranker::test_empty_chunks PASSED  [ 70%]
... (remaining PASSED/XFAIL lines omitted) ...


... (remaining PASSED/XFAIL lines omitted) ...
  /home/user/Documents/github/codepath/pathreview-ai301-fa26-s3/core/config.py:7: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.13/migration/
    class Settings(BaseSettings):

tests/unit/test_semantic_chunker.py::TestSemanticChunker::test_empty_input_returns_empty_list
  /home/user/Documents/github/codepath/pathreview-ai301-fa26-s3/.venv/lib/python3.14/site-packages/requests/certs.py:15: RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited
    from certifi import where
  Enable tracemalloc to get traceback where the object was allocated.
  See https://docs.pytest.org/en/stable/how-to/capture-warnings.html#resource-warnings for more info.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 391 passed, 53 xfailed, 2 warnings in 7.04s ==================
exit: 0
```

## Diff

Code diff only: the two code files, `rag/retriever/reranker.py` and
`tests/unit/test_reranking.py`, as of commit `e056a7f`. `plan.md` and `comment.md`
are added in a separate docs commit and are not shown here.

```diff
$ git diff main...e056a7f
diff --git a/rag/retriever/reranker.py b/rag/retriever/reranker.py
new file mode 100644
index 0000000..e47e5c0
--- /dev/null
+++ b/rag/retriever/reranker.py
@@ -0,0 +1,94 @@
+"""Optional LLM re-ranking of retrieved chunks."""
+
+import json
+
+import structlog
+
+logger = structlog.get_logger()
+
+MIN_SCORE = 0
+MAX_SCORE = 10
+
+SYSTEM_PROMPT = (
+    "You rate how relevant each numbered text chunk is to a query. "
+    f"Give each chunk an integer from {MIN_SCORE} (irrelevant) to {MAX_SCORE} (directly answers "
+    "the query). Reply with only a JSON list of integers, one per chunk, in the same order "
+    "as the chunks. No other text."
+)
+
+USER_PROMPT = "Query: {query}\n\nChunks:\n{chunks}"
+
+
+class LLMReranker:
+    """Re-rank retrieved chunks by asking an LLM to score each one against the query."""
+
+    def __init__(self, client, model: str, enabled: bool = False):
+        """Initialize reranker.
+
+        Args:
+            client: OpenAI-compatible client (e.g. openai.OpenAI pointed at Ollama)
+            model: Model name to score with
+            enabled: When False, rerank() returns chunks[:top_k] without calling the LLM
+        """
+        self.client = client
+        self.model = model
+        self.enabled = enabled
+
+    def rerank(self, query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
+        """Re-rank chunks by LLM relevance score.
+
+        Args:
+            query: Query text
+            chunks: Chunks as returned by HybridRetriever.retrieve()
+            top_k: Maximum chunks to return
+
+        Returns:
+            Up to top_k chunks, highest LLM score first. Ties keep the input order. Only
+            "score" is replaced. If disabled or the LLM reply is unusable, the first top_k
+            chunks are returned in their original order.
+        """
+        if not self.enabled or not chunks:
+            return chunks[:top_k]
+
+        scores = self._score(query, chunks)
+        if scores is None:
+            return chunks[:top_k]
+
+        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
+        reranked = [{**chunks[i], "score": scores[i]} for i in order]
+
+        logger.info("rerank_complete", input_count=len(chunks), top_k=top_k)
+        return reranked[:top_k]
+
+    def _score(self, query: str, chunks: list[dict]) -> list[int] | None:
+        """Ask the LLM for one integer score per chunk, or None if the reply is unusable."""
+        listing = "\n".join(f"[{i + 1}] {chunk.get('text', '')}" for i, chunk in enumerate(chunks))
+        try:
+            response = self.client.chat.completions.create(
+                model=self.model,
+                temperature=0,
+                messages=[
+                    {"role": "system", "content": SYSTEM_PROMPT},
+                    {
+                        "role": "user",
+                        "content": USER_PROMPT.format(query=query, chunks=listing),
+                    },
+                ],
+            )
+            scores = json.loads(response.choices[0].message.content)
+        except Exception as e:
+            logger.warning("rerank_failed", error=str(e))
+            return None
+
+        valid = (
+            isinstance(scores, list)
+            and len(scores) == len(chunks)
+            and all(
+                isinstance(s, int) and not isinstance(s, bool) and MIN_SCORE <= s <= MAX_SCORE
+                for s in scores
+            )
+        )
+        if not valid:
+            logger.warning("rerank_bad_reply", expected=len(chunks))
+            return None
+        return scores
diff --git a/tests/unit/test_reranking.py b/tests/unit/test_reranking.py
new file mode 100644
index 0000000..3b318de
--- /dev/null
+++ b/tests/unit/test_reranking.py
@@ -0,0 +1,111 @@
+"""Tests for reranker.py"""
+
+import json
+from unittest.mock import MagicMock
+
+import pytest
+
+from rag.retriever.reranker import LLMReranker
+
+
+def make_client(reply: str) -> MagicMock:
+    """Mock OpenAI-compatible client whose completion content is `reply`."""
+    client = MagicMock()
+    client.chat.completions.create.return_value.choices = [MagicMock()]
+    client.chat.completions.create.return_value.choices[0].message.content = reply
+    return client
+
+
+def make_chunks(n: int = 3) -> list[dict]:
+    return [
+        {
+            "id": f"c{i}",
+            "text": f"chunk text {i}",
+            "metadata": {"source_id": "s", "chunk_index": i, "section": "skills"},
+            "score": 0.5,
+            "vector_score": 0.4,
+            "keyword_score": 0.6,
+        }
+        for i in range(1, n + 1)
+    ]
+
+
+def ids(chunks: list[dict]) -> list[str]:
+    return [c["id"] for c in chunks]
+
+
+@pytest.mark.unit
+class TestLLMReranker:
+    """Test suite for LLMReranker."""
+
+    def test_orders_by_llm_score_descending(self):
+        reranker = LLMReranker(make_client(json.dumps([2, 9, 5])), "m", enabled=True)
+        result = reranker.rerank("q", make_chunks())
+        assert ids(result) == ["c2", "c3", "c1"]
+        assert [c["score"] for c in result] == [9, 5, 2]
+
+    def test_top_k_caps_to_highest_scoring(self):
+        reranker = LLMReranker(make_client(json.dumps([2, 9, 5])), "m", enabled=True)
+        result = reranker.rerank("q", make_chunks(), top_k=2)
+        assert ids(result) == ["c2", "c3"]
+
+    def test_disabled_passes_through_without_calling_llm(self):
+        client = make_client(json.dumps([2, 9, 5]))
+        reranker = LLMReranker(client, "m")
+        result = reranker.rerank("q", make_chunks())
+        assert ids(result) == ["c1", "c2", "c3"]
+        client.chat.completions.create.assert_not_called()
+
+    def test_disabled_still_applies_top_k(self):
+        reranker = LLMReranker(make_client("[]"), "m")
+        assert ids(reranker.rerank("q", make_chunks(), top_k=2)) == ["c1", "c2"]
+
+    def test_dict_shape_and_other_scores_unchanged(self):
+        reranker = LLMReranker(make_client(json.dumps([2, 9, 5])), "m", enabled=True)
+        original = make_chunks()
+        result = reranker.rerank("q", original)
+        for chunk in result:
+            assert set(chunk) == {
+                "id",
+                "text",
+                "metadata",
+                "score",
+                "vector_score",
+                "keyword_score",
+            }
+            assert chunk["vector_score"] == 0.4
+            assert chunk["keyword_score"] == 0.6
+        assert [c["score"] for c in original] == [0.5, 0.5, 0.5]
+
+    @pytest.mark.parametrize(
+        "reply",
+        [
+            "not json",
+            json.dumps([1, 2]),
+            json.dumps([1, 2, 3, 4]),
+            json.dumps([1, 2, 11]),
+            json.dumps([1, 2, -1]),
+            json.dumps([1, 2, 3.5]),
+            json.dumps([1, 2, True]),
+            json.dumps({"scores": [1, 2, 3]}),
+        ],
+    )
+    def test_bad_reply_returns_original_order(self, reply):
+        reranker = LLMReranker(make_client(reply), "m", enabled=True)
+        assert ids(reranker.rerank("q", make_chunks())) == ["c1", "c2", "c3"]
+
+    def test_client_error_returns_original_order(self):
+        client = MagicMock()
+        client.chat.completions.create.side_effect = RuntimeError("ollama down")
+        reranker = LLMReranker(client, "m", enabled=True)
+        assert ids(reranker.rerank("q", make_chunks())) == ["c1", "c2", "c3"]
+
+    def test_ties_keep_retriever_order(self):
+        reranker = LLMReranker(make_client(json.dumps([5, 5, 7])), "m", enabled=True)
+        assert ids(reranker.rerank("q", make_chunks())) == ["c3", "c1", "c2"]
+
+    def test_empty_chunks(self):
+        client = make_client("[]")
+        reranker = LLMReranker(client, "m", enabled=True)
+        assert reranker.rerank("q", []) == []
+        client.chat.completions.create.assert_not_called()
```

## AI Use Statement

I'm using Claude Code to read the repo, check this plan against a rubric, and
help with debugging. I review everything it produces before posting.

## Deviations

`test_disabled_still_applies_top_k` and `test_client_error_returns_original_order`
go a bit beyond the plan.  Otherwise, it's good to go.

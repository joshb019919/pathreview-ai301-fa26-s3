"""Tests for reranker.py"""

import json
from unittest.mock import MagicMock

import pytest

from rag.retriever.reranker import LLMReranker


def make_client(reply: str) -> MagicMock:
    """Mock OpenAI-compatible client whose completion content is `reply`."""
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock()]
    client.chat.completions.create.return_value.choices[0].message.content = reply
    return client


def make_chunks(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"c{i}",
            "text": f"chunk text {i}",
            "metadata": {"source_id": "s", "chunk_index": i, "section": "skills"},
            "score": 0.5,
            "vector_score": 0.4,
            "keyword_score": 0.6,
        }
        for i in range(1, n + 1)
    ]


def ids(chunks: list[dict]) -> list[str]:
    return [c["id"] for c in chunks]


@pytest.mark.unit
class TestLLMReranker:
    """Test suite for LLMReranker."""

    def test_orders_by_llm_score_descending(self):
        reranker = LLMReranker(make_client(json.dumps([2, 9, 5])), "m", enabled=True)
        result = reranker.rerank("q", make_chunks())
        assert ids(result) == ["c2", "c3", "c1"]
        assert [c["score"] for c in result] == [9, 5, 2]

    def test_top_k_caps_to_highest_scoring(self):
        reranker = LLMReranker(make_client(json.dumps([2, 9, 5])), "m", enabled=True)
        result = reranker.rerank("q", make_chunks(), top_k=2)
        assert ids(result) == ["c2", "c3"]

    def test_disabled_passes_through_without_calling_llm(self):
        client = make_client(json.dumps([2, 9, 5]))
        reranker = LLMReranker(client, "m")
        result = reranker.rerank("q", make_chunks())
        assert ids(result) == ["c1", "c2", "c3"]
        client.chat.completions.create.assert_not_called()

    def test_disabled_still_applies_top_k(self):
        reranker = LLMReranker(make_client("[]"), "m")
        assert ids(reranker.rerank("q", make_chunks(), top_k=2)) == ["c1", "c2"]

    def test_dict_shape_and_other_scores_unchanged(self):
        reranker = LLMReranker(make_client(json.dumps([2, 9, 5])), "m", enabled=True)
        original = make_chunks()
        result = reranker.rerank("q", original)
        for chunk in result:
            assert set(chunk) == {
                "id",
                "text",
                "metadata",
                "score",
                "vector_score",
                "keyword_score",
            }
            assert chunk["vector_score"] == 0.4
            assert chunk["keyword_score"] == 0.6
        assert [c["score"] for c in original] == [0.5, 0.5, 0.5]

    @pytest.mark.parametrize(
        "reply",
        [
            "not json",
            json.dumps([1, 2]),
            json.dumps([1, 2, 3, 4]),
            json.dumps([1, 2, 11]),
            json.dumps([1, 2, -1]),
            json.dumps([1, 2, 3.5]),
            json.dumps([1, 2, True]),
            json.dumps({"scores": [1, 2, 3]}),
        ],
    )
    def test_bad_reply_returns_original_order(self, reply):
        reranker = LLMReranker(make_client(reply), "m", enabled=True)
        assert ids(reranker.rerank("q", make_chunks())) == ["c1", "c2", "c3"]

    def test_client_error_returns_original_order(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("ollama down")
        reranker = LLMReranker(client, "m", enabled=True)
        assert ids(reranker.rerank("q", make_chunks())) == ["c1", "c2", "c3"]

    def test_ties_keep_retriever_order(self):
        reranker = LLMReranker(make_client(json.dumps([5, 5, 7])), "m", enabled=True)
        assert ids(reranker.rerank("q", make_chunks())) == ["c3", "c1", "c2"]

    def test_empty_chunks(self):
        client = make_client("[]")
        reranker = LLMReranker(client, "m", enabled=True)
        assert reranker.rerank("q", []) == []
        client.chat.completions.create.assert_not_called()

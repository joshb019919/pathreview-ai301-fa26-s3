"""Optional LLM re-ranking of retrieved chunks."""

import json

import structlog

logger = structlog.get_logger()

MIN_SCORE = 0
MAX_SCORE = 10

SYSTEM_PROMPT = (
    "You rate how relevant each numbered text chunk is to a query. "
    f"Give each chunk an integer from {MIN_SCORE} (irrelevant) to {MAX_SCORE} (directly answers "
    "the query). Reply with only a JSON list of integers, one per chunk, in the same order "
    "as the chunks. No other text."
)

USER_PROMPT = "Query: {query}\n\nChunks:\n{chunks}"


class LLMReranker:
    """Re-rank retrieved chunks by asking an LLM to score each one against the query."""

    def __init__(self, client, model: str, enabled: bool = False):
        """Initialize reranker.

        Args:
            client: OpenAI-compatible client (e.g. openai.OpenAI pointed at Ollama)
            model: Model name to score with
            enabled: When False, rerank() returns chunks[:top_k] without calling the LLM
        """
        self.client = client
        self.model = model
        self.enabled = enabled

    def rerank(self, query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
        """Re-rank chunks by LLM relevance score.

        Args:
            query: Query text
            chunks: Chunks as returned by HybridRetriever.retrieve()
            top_k: Maximum chunks to return

        Returns:
            Up to top_k chunks, highest LLM score first. Ties keep the input order. Only
            "score" is replaced. If disabled or the LLM reply is unusable, the first top_k
            chunks are returned in their original order.
        """
        if not self.enabled or not chunks:
            return chunks[:top_k]

        scores = self._score(query, chunks)
        if scores is None:
            return chunks[:top_k]

        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        reranked = [{**chunks[i], "score": scores[i]} for i in order]

        logger.info("rerank_complete", input_count=len(chunks), top_k=top_k)
        return reranked[:top_k]

    def _score(self, query: str, chunks: list[dict]) -> list[int] | None:
        """Ask the LLM for one integer score per chunk, or None if the reply is unusable."""
        listing = "\n".join(f"[{i + 1}] {chunk.get('text', '')}" for i, chunk in enumerate(chunks))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(query=query, chunks=listing),
                    },
                ],
            )
            scores = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning("rerank_failed", error=str(e))
            return None

        valid = (
            isinstance(scores, list)
            and len(scores) == len(chunks)
            and all(
                isinstance(s, int) and not isinstance(s, bool) and MIN_SCORE <= s <= MAX_SCORE
                for s in scores
            )
        )
        if not valid:
            logger.warning("rerank_bad_reply", expected=len(chunks))
            return None
        return scores

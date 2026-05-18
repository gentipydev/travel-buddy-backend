"""Retrieval-augmented generation pipeline.

Public API:
  - `retrieve_chunks(query, db, history)` — full pipeline (route → embed →
    rerank), returns `(top_n_chunks, top_rerank_score)`
  - `retrieve_context(query, db, history)` — convenience join of chunk
    contents into a single string
  - `needs_retrieval(message)` — the routing heuristic (skip greetings,
    yes/no, thanks)
  - `search_chunks`, `rerank_chunks` — lower-level building blocks
  - `detect_place(message, db)` + `to_card_dict(place)` — place-card
    detection used by /chat/stream
"""
from app.rag.embedding import search_chunks
from app.rag.place_detection import detect_place, to_card_dict
from app.rag.reranking import rerank_chunks
from app.rag.retrieval import needs_retrieval, retrieve_chunks, retrieve_context

__all__ = [
    "detect_place",
    "needs_retrieval",
    "rerank_chunks",
    "retrieve_chunks",
    "retrieve_context",
    "search_chunks",
    "to_card_dict",
]

"""Unit tests for the contextual-retrieval helpers (app/rag/contextual.py).

Pure functions — no DB, no Cohere. They build the metadata header that
`chunk_places.py` / `seed_eval_db.py` store in `Chunk.context` and that
`embed_chunks.py` prepends to the embedded text.
"""
from app.rag.contextual import context_header, embedding_input

# ---------------------------------------------------------------------- #
# context_header                                                         #
# ---------------------------------------------------------------------- #


def test_context_header_full_fields():
    assert context_header(
        "Berat", "Berat", "Berat County", "city", ["unesco", "ottoman"]
    ) == "Berat — city in Berat, Berat County, Albania. Themes: unesco, ottoman."


def test_context_header_omits_missing_region():
    assert (
        context_header("Theth", "Theth", None, "village", ["alps"])
        == "Theth — village in Theth, Albania. Themes: alps."
    )


def test_context_header_without_category():
    assert (
        context_header("X", "Y", "Z", None, ["t"])
        == "X — in Y, Z, Albania. Themes: t."
    )


def test_context_header_without_tags():
    assert (
        context_header("Furgons", None, None, "transport", None)
        == "Furgons — transport in Albania."
    )


def test_context_header_drops_blank_tags():
    assert (
        context_header("A", "B", "C", "d", ["", "  ", "x"])
        == "A — d in B, C, Albania. Themes: x."
    )


def test_context_header_empty_name_returns_empty():
    assert context_header("", "B", "C", "d", ["x"]) == ""
    assert context_header("   ", "B", "C", "d", ["x"]) == ""


# ---------------------------------------------------------------------- #
# embedding_input                                                        #
# ---------------------------------------------------------------------- #


def test_embedding_input_joins_context_and_content():
    assert embedding_input("HDR", "body text") == "HDR\n\nbody text"


def test_embedding_input_falls_back_to_content_when_no_context():
    assert embedding_input("", "body") == "body"
    assert embedding_input(None, "body") == "body"
    assert embedding_input("   ", "body") == "body"

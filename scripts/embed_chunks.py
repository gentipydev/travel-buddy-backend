"""Embed every Chunk.content via Cohere and store the 1024-dim vector in
Chunk.embedding. Picks up only chunks where embedding IS NULL, so re-running
is cheap. `scripts/chunk_places.py` already clears the embedding when its
content changes, which means re-running this script naturally re-embeds
edited chunks.

Usage (from the project root, with COHERE_API_KEY set in .env):
    python -m scripts.embed_chunks
"""
import asyncio

import cohere
from sqlalchemy import select

from app.config import settings
from app.database import Chunk, async_session


BATCH_SIZE = 96  # Cohere embed API accepts up to 96 inputs per call
INPUT_TYPE = "search_document"  # v3 models require this; queries use "search_query"


async def embed():
    if not settings.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is not set — add it to .env")

    async with async_session() as db:
        pending = (
            await db.execute(select(Chunk).where(Chunk.embedding.is_(None)))
        ).scalars().all()

        if not pending:
            print("embed_chunks: nothing to embed (all chunks already have embeddings)")
            return

        print(
            f"embed_chunks: embedding {len(pending)} chunks "
            f"via {settings.embedding_model}"
        )

        embedded = 0
        async with cohere.AsyncClientV2(api_key=settings.cohere_api_key) as co:
            for i in range(0, len(pending), BATCH_SIZE):
                batch = pending[i : i + BATCH_SIZE]
                response = await co.embed(
                    model=settings.embedding_model,
                    texts=[c.content for c in batch],
                    input_type=INPUT_TYPE,
                    embedding_types=["float"],
                )
                vectors = response.embeddings.float_
                for chunk, vector in zip(batch, vectors):
                    chunk.embedding = vector
                    embedded += 1
                print(f"  batch {i // BATCH_SIZE + 1}: +{len(batch)} embedded")

        await db.commit()
        print(f"embed_chunks: embedded={embedded}")


if __name__ == "__main__":
    asyncio.run(embed())

"""Ingest: load PDFs -> chunk -> embed (HuggingFace) -> store in ChromaDB (file-wise)."""
import gc
from functools import lru_cache

import chromadb
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config


class Embedder:
    """HuggingFace embedding model (all-MiniLM-L6-v2) run with ONNX: no PyTorch, low memory."""

    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name, threads=1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=8)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.embed([text]))).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embedder:
    return Embedder(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def load_pdf(path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for one PDF."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def ingest_file(path) -> int:
    """(Re)index a single PDF. Old chunks of the same file are replaced."""
    name = path.name
    col = get_collection()
    col.delete(where={"source": name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    docs, metas, ids = [], [], []
    for page_no, text in load_pdf(path):
        for j, chunk in enumerate(splitter.split_text(text)):
            docs.append(chunk)
            metas.append({"source": name, "page": page_no})
            ids.append(f"{name}::p{page_no}::c{j}")

    emb = get_embeddings()
    for i in range(0, len(docs), 16):
        batch = docs[i : i + 16]
        col.add(
            ids=ids[i : i + 16],
            documents=batch,
            metadatas=metas[i : i + 16],
            embeddings=emb.embed_documents(batch),
        )
    count = len(docs)
    del docs, metas, ids
    gc.collect()
    return count


def ingest_all(force: bool = False) -> dict:
    """Index every PDF in data/. Skips files already indexed unless force=True."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    col = get_collection()
    result = {}
    for pdf in sorted(config.DATA_DIR.glob("*.pdf")):
        if not force and col.get(where={"source": pdf.name}, limit=1)["ids"]:
            result[pdf.name] = "skipped (already indexed)"
            continue
        try:
            result[pdf.name] = f"{ingest_file(pdf)} chunks"
        except Exception as e:  # keep going if one PDF is broken
            result[pdf.name] = f"failed: {e}"
    return result


def list_files() -> dict:
    """{filename: chunk_count} for everything in the vector DB."""
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for m in metas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return dict(sorted(counts.items()))
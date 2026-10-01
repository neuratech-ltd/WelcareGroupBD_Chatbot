"""Ingest: load PDFs -> chunk -> embed (HuggingFace) -> store in ChromaDB (file-wise)."""
from functools import lru_cache

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


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
    for i in range(0, len(docs), 64):
        batch = docs[i : i + 64]
        col.add(
            ids=ids[i : i + 64],
            documents=batch,
            metadatas=metas[i : i + 64],
            embeddings=emb.embed_documents(batch),
        )
    return len(docs)


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

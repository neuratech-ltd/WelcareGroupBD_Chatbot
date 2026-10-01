"""Retrieval + generation with Groq."""
import re

from groq import Groq

from app import config
from app.ingest import get_collection, get_embeddings

NOT_FOUND = "I couldn't find this in the company documents."

SYSTEM_PROMPT = f"""You are the internal assistant of {config.COMPANY_NAME}.
Answer the question using ONLY the context provided.
If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
Do not invent facts.
Never mention file names, document names, page numbers, or phrases like "the context" or "the provided text". Just answer naturally.
Write in a friendly, conversational tone, like a helpful assistant.
Format with Markdown: a short direct answer first, then bullet points for lists,
**bold** for key names and numbers, and a table when comparing items.
Keep it clear and not too long."""


GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "heyy", "hey there", "hello there",
    "hi there", "yo", "salam", "assalamualaikum", "good morning",
    "good afternoon", "good evening",
}
THANKS = {"thanks", "thank you", "thx", "thanks a lot", "ok", "okay", "ok thanks"}
BYE = {"bye", "goodbye", "see you"}


def small_talk(question: str) -> str | None:
    """Fixed replies for greetings, thanks and goodbyes (no Groq call)."""
    text = " ".join(re.sub(r"[^a-z\s]", "", question.lower()).split())
    if text in GREETINGS:
        return f"Hello! Welcome to {config.COMPANY_NAME}. Ask me anything about our documents."
    if text in THANKS:
        return "You're welcome! Ask another question anytime."
    if text in BYE:
        return "Goodbye! Have a great day."
    return None


def answer(question: str, source: str | None = None) -> dict:
    reply = small_talk(question)
    if reply:
        return {"answer": reply}

    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    query_vec = get_embeddings().embed_query(question)
    res = get_collection().query(
        query_embeddings=[query_vec],
        n_results=config.TOP_K,
        where={"source": source} if source else None,
        include=["documents", "metadatas"],
    )
    docs, metas = res["documents"][0], res["metadatas"][0]
    if not docs:
        return {"answer": NOT_FOUND}

    context = "\n\n---\n\n".join(docs)  # file names are never sent to the model
    client = Groq(api_key=config.GROQ_API_KEY)
    chat = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = (chat.choices[0].message.content or "").strip()
    if NOT_FOUND.lower() in text.lower():
        return {"answer": NOT_FOUND}
    return {"answer": text}

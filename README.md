# Welcare Group BD - Document Assistant (RAG Chatbot)

A chatbot that answers questions from your company PDFs.

**Stack:** FastAPI, ChromaDB (vector database), HuggingFace embeddings (`all-MiniLM-L6-v2`, run with ONNX so it uses little memory), Groq (LLM), and a small light-themed chat page. Everything runs in one Docker container.

## How it works

1. **Ingest:** every PDF in the `data/` folder is read page by page.
2. **Chunking:** the text is split into overlapping chunks (800 characters, 150 overlap).
3. **Embedding:** each chunk is turned into a vector with the HuggingFace MiniLM model.
4. **Vector database:** chunks are stored in ChromaDB, file by file (re-indexing a file replaces its old chunks).
5. **Question:** the question is embedded, the 4 most similar chunks are found, and Groq writes the answer from them.
6. Greetings such as "hey" or "thanks" get a fixed reply and do not call Groq.
7. Users only see the answer. File names and page numbers are never shown.

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   ├── config.py        # settings (read from environment variables)
│   ├── ingest.py        # load PDFs, chunk, embed, store in ChromaDB
│   ├── rag.py           # retrieval + Groq answer
│   └── static/index.html  # chat page
├── data/                # PUT YOUR PDF FILES HERE
├── chroma_db/           # vector database (created automatically)
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup (run on your computer)

1. Put your PDF files directly inside the `data/` folder (not in sub-folders).
2. Copy `.env.example` to `.env` and fill it in:
   ```
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=openai/gpt-oss-120b
   COMPANY_NAME=Welcare Group BD
   ADMIN_KEY=a_long_random_password
   ```
3. Start it:
   ```
   docker compose up --build
   ```
4. Open http://localhost:8000 and ask a question.

PDFs are indexed automatically when the app starts. Files that are already indexed are skipped.

## Adding or changing PDFs

- **New PDF:** copy it into `data/`, then restart: `docker compose restart`.
- **Changed PDF with the same file name:** re-index it (needs your admin key):
  ```
  curl -X POST "http://localhost:8000/ingest?force=true" -H "x-admin-key: YOUR_ADMIN_KEY"
  ```
  On Windows PowerShell use `curl.exe` instead of `curl`.
- **Deleted PDF:** its data stays in the database. To remove it, stop the app, delete everything inside `chroma_db/` except `.gitkeep`, and start again.

## Deploy on Render

1. Put the project on GitHub in a **private** repository, with your PDFs inside `data/`. The `.env` file is git-ignored, so your keys are not uploaded.
2. In Render: New, then Web Service, connect the repository, and choose the **Docker** runtime.
3. Under Environment, add:
   - `GROQ_API_KEY`
   - `GROQ_MODEL` = `openai/gpt-oss-120b`
   - `ADMIN_KEY` (a long random value)
   - `COMPANY_NAME` (optional)
4. Deploy. The PDFs are indexed while the image is built.
5. To update PDFs or code later: commit, push to GitHub, and Render redeploys automatically.

Notes for the Render free plan:
- It has 512 MB of memory, which is why the lightweight ONNX embedding is used.
- The service sleeps after inactivity, and the first visit afterwards can take about a minute.
- The disk is temporary. The index is rebuilt from `data/` on every deploy.

## Endpoints

| Endpoint | Access | Purpose |
|---|---|---|
| `GET /` | public | chat page |
| `POST /ask` | public | `{"question": "..."}` returns `{"answer": "..."}` |
| `GET /health` | public | shows whether indexing has finished |
| `GET /files` | admin key | list indexed files and chunk counts |
| `POST /ingest?force=false` | admin key | index PDFs from `data/` |

Admin endpoints need the header `x-admin-key: YOUR_ADMIN_KEY`. The `/docs` page is turned off.

## Settings (environment variables)

| Name | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | your Groq key (required) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `COMPANY_NAME` | Welcare Group BD | shown in the page and prompt |
| `ADMIN_KEY` | none | password for `/files` and `/ingest` |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | chunking settings |
| `TOP_K` | 4 | chunks sent to the model per question |

## Troubleshooting

- **"I couldn't find this in the company documents":** the answer is not in your PDFs, or no PDFs are indexed. Check that the PDFs are in `data/` and contain selectable text (scanned PDFs need OCR).
- **Groq error "model not found":** your key may not have access to that model. Set a different `GROQ_MODEL` from the list in your Groq console.
- **"Documents are still being indexed":** wait a minute and try again.
- **502 on Render:** wait a minute if the service was asleep. If it keeps restarting, check the Events page for an out-of-memory message.

## Security

- Keep the GitHub repository private, since the PDFs are inside it.
- Never commit `.env`. If a key is ever shown publicly, create a new one.
- Anyone with the link can use the chat page. Add a login before sharing it with many people.
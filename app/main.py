import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, ingest, rag

state = {"ready": False, "error": None}


def _startup_ingest():
    """Index PDFs in the background so the server binds its port immediately."""
    try:
        ingest.get_embeddings()
        ingest.ingest_all()
    except Exception as e:
        state["error"] = str(e)
    finally:
        state["ready"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_startup_ingest, daemon=True).start()
    yield


app = FastAPI(
    title=f"{config.COMPANY_NAME} - Document Assistant",
    lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None,  # hide public API docs
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    file: str | None = None


@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "ready": state["ready"], "error": state["error"]}


@app.get("/config")
def public_config():
    return {"company": config.COMPANY_NAME}


def require_admin(key: str):
    if not config.ADMIN_KEY or key != config.ADMIN_KEY:
        raise HTTPException(401, "Admin key required.")


@app.get("/files")
def files(x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)  # file names are private: admin only
    return {"files": ingest.list_files()}


@app.post("/ingest")
def run_ingest(force: bool = False, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    return {"result": ingest.ingest_all(force=force)}


@app.post("/ask")
def ask(req: AskRequest):
    if not state["ready"]:
        raise HTTPException(503, "Documents are still being indexed. Try again in a minute.")
    try:
        return rag.answer(req.question.strip(), req.file)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not get an answer: {e}")
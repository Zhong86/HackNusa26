from dotenv import load_dotenv

load_dotenv()  # baca .env sebelum modul lain (llm, embeddings) membaca env var

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from db import init_db

init_db()

app = FastAPI(title="Sentinel Loop — Layer 2 Agent", version="0.1.0")
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

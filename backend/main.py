from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="Sentinel Loop — Layer 2 Agent", version="0.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
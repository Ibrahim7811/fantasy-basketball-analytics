from fastapi import FastAPI

app = FastAPI(title="Fantasy Basketball Analyzer")
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
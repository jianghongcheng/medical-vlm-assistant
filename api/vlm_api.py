"""
Medical VLM Report Assistant API
FastAPI backend integrating Stanford Merlin + RAG + GPT-4o
"""

import os
import time
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Load env
for line in Path('.env').read_text().splitlines():
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

# Init pipeline
from merlin_pipeline import (
    init_openai, init_qdrant, index_sample_reports, run_full_pipeline
)

app = FastAPI(
    title="Medical VLM Report Assistant",
    description="CT scan analysis using Stanford Merlin (Nature 2026) + RAG + GPT-4o",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Init DB
DB_PATH = "vlm_monitoring.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_description TEXT,
            risk_level TEXT,
            latency_ms REAL,
            embedding_dim INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Startup
@app.on_event("startup")
async def startup():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    init_openai(api_key)
    init_qdrant()
    index_sample_reports()
    init_db()
    print("Medical VLM Assistant ready!")

# ── Models ───────────────────────────────────────────
class AnalysisRequest(BaseModel):
    patient_description: str = "Abdominal CT scan for evaluation"

class TextQueryRequest(BaseModel):
    query: str

# ── Endpoints ────────────────────────────────────────
@app.get("/")
async def root():
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Medical VLM Report Assistant", "docs": "/docs"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "Stanford Merlin (Nature 2026)",
        "capabilities": [
            "CT ImageEmbedding (2048-dim)",
            "RAG clinical report retrieval",
            "GPT-4o structured report generation"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze")
async def analyze_ct(
    file: UploadFile = File(...),
    patient_description: str = "CT scan analysis"
):
    """
    Upload a CT scan (.nii.gz) and get a structured clinical report.
    Pipeline: Merlin ImageEmbedding → RAG → GPT-4o report generation
    """
    if not (file.filename.endswith('.nii.gz') or file.filename.endswith('.nii')):
        raise HTTPException(
            status_code=400,
            detail="Only NIfTI files (.nii or .nii.gz) are supported"
        )

    # Save uploaded file
    with tempfile.NamedTemporaryFile(
        suffix='.nii.gz' if file.filename.endswith('.gz') else '.nii',
        delete=False
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_full_pipeline(tmp_path, patient_description)

        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO analyses (timestamp, patient_description, risk_level, latency_ms, embedding_dim) VALUES (?,?,?,?,?)",
            (
                datetime.now().isoformat(),
                patient_description[:200],
                result.get("risk_level", "UNKNOWN"),
                result.get("latency_ms", 0),
                result.get("embedding_dim", 0)
            )
        )
        conn.commit()
        conn.close()

        return result

    finally:
        os.unlink(tmp_path)

@app.post("/analyze/demo")
async def analyze_demo(request: AnalysisRequest):
    """
    Run analysis on the built-in demo CT scan (no upload needed).
    Uses Merlin's sample abdominal CT data.
    """
    demo_path = "abct_data/image1.nii.gz"

    if not Path(demo_path).exists():
        from merlin.data import download_sample_data
        download_sample_data("abct_data")

    result = run_full_pipeline(demo_path, request.patient_description)

    # Log to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO analyses (timestamp, patient_description, risk_level, latency_ms, embedding_dim) VALUES (?,?,?,?,?)",
        (
            datetime.now().isoformat(),
            request.patient_description[:200],
            result.get("risk_level", "UNKNOWN"),
            result.get("latency_ms", 0),
            result.get("embedding_dim", 0)
        )
    )
    conn.commit()
    conn.close()

    return result

@app.post("/query")
async def query_knowledge_base(request: TextQueryRequest):
    """Query the radiology report knowledge base using RAG"""
    from merlin_pipeline import retrieve_similar_reports
    reports = retrieve_similar_reports(request.query, top_k=3)
    return {
        "query": request.query,
        "results": reports,
        "count": len(reports)
    }

@app.get("/metrics")
async def metrics():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT
                COUNT(*) as total,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END) as high_risk,
                SUM(CASE WHEN risk_level='MODERATE' THEN 1 ELSE 0 END) as moderate_risk,
                SUM(CASE WHEN risk_level='LOW' THEN 1 ELSE 0 END) as low_risk
            FROM analyses
        """).fetchone()
        conn.close()
        return {
            "total_analyses": rows[0] or 0,
            "avg_latency_ms": round(rows[1] or 0, 1),
            "risk_distribution": {
                "HIGH": rows[2] or 0,
                "MODERATE": rows[3] or 0,
                "LOW": rows[4] or 0
            },
            "model": "Stanford Merlin (Nature 2026) + GPT-4o",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Medical VLM Report Assistant...")
    print("Model: Stanford Merlin (Nature 2026) + GPT-4o")
    print("Docs: http://localhost:8001/docs")
    print("App:  http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

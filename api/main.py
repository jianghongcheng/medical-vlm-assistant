"""
Medical VLM CT Report Generation System
FastAPI Main Entry
"""
import os
import sys
import sqlite3
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, '/media/max/b/LLM/medical-vlm-assistant')
from api.core import analyze_ct, load_vlm, load_clip
from qdrant_client import QdrantClient

# 全局Qdrant客户端
qdrant = QdrantClient(host='localhost', port=6333)
from qdrant_client import QdrantClient

# 全局Qdrant客户端
qdrant = QdrantClient(host='localhost', port=6333)

app = FastAPI(
    title="Medical VLM CT Report Generation",
    description="""
    CT Report Generation System based on Med3DVLM (IEEE JBHI 2025)
    
    Architecture:
    - Stage 1: Structured Perception (VQA, 79.95% accuracy)
    - Stage 2: Image RAG (DCFormer-SigLIP, Recall@1=61%)
    - Stage 3: Report Generation + Uncertainty (METEOR=36.42%)
    - Stage 4: Safety Layer (Human-in-the-loop, always DRAFT)
    
    ⚠️ FOR RESEARCH USE ONLY. Not for clinical diagnosis.
    """,
    version="2.0.0"
)

# 挂载前端静态文件
app.mount("/static", StaticFiles(directory="/media/max/b/LLM/medical-vlm-assistant/frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# SQLite监控
DB_PATH = "/tmp/medical_vlm_monitor.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            filename    TEXT,
            latency_s   REAL,
            confidence  TEXT,
            risk_level  TEXT,
            stage1_s    REAL,
            stage3_s    REAL
        )
    """)
    conn.commit()
    conn.close()

def log_result(filename, result):
    try:
        stages = result.get("pipeline_stages", {})
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO analyses VALUES (NULL,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(),
                filename,
                result.get("total_latency_seconds", 0),
                result.get("confidence", "UNKNOWN"),
                result.get("risk_level", "UNKNOWN"),
                stages.get("structured_perception", {}).get("latency_seconds", 0),
                stages.get("report_generation", {}).get("latency_seconds", 0),
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Log failed: {e}")

init_db()

@app.on_event("startup")
async def startup():
    """预加载所有模型"""
    print("=" * 50)
    print("Medical VLM System Starting...")
    print("Loading VLM model...")
    load_vlm()
    print("Loading CLIP model...")
    load_clip()
    print("All models ready!")
    print("=" * 50)

@app.get("/ui")
def ui():
    return FileResponse("/media/max/b/LLM/medical-vlm-assistant/frontend/index.html")

@app.get("/")
def root():
    return {
        "service": "Medical VLM CT Report Generation",
        "model": "Med3DVLM (IEEE JBHI 2025)",
        "architecture": {
            "encoder": "DCFormer (decomposed 3D convolutions)",
            "alignment": "SigLIP (sigmoid contrastive loss)",
            "llm": "Qwen2.5-7B",
            "retrieval": "DCFormer-SigLIP + Qdrant image RAG"
        },
        "performance": {
            "retrieval_recall_at_1": "61.00% (2000 candidates)",
            "report_meteor": "36.42%",
            "vqa_accuracy": "79.95%",
            "inference_latency": "~1.0s on RTX 3090"
        },
        "safety": {
            "review_required": True,
            "status": "ALWAYS DRAFT",
            "disclaimer": "FOR RESEARCH USE ONLY"
        },
        "endpoints": {
            "POST /analyze": "Full CT analysis pipeline",
            "GET  /stats":   "Performance statistics",
            "GET  /health":  "System health check",
            "GET  /docs":    "API documentation"
        }
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Complete CT Analysis Pipeline
    
    Input:  NIfTI CT file (.nii or .nii.gz)
    Output: Structured report with confidence and safety layer
    
    Pipeline stages:
    1. Structured Perception (VQA-based)
    2. Image RAG (DCFormer-SigLIP)
    3. Report Generation + Uncertainty
    4. Safety Layer (Human-in-the-loop)
    """
    # 验证文件格式
    if not (file.filename.endswith('.nii') or
            file.filename.endswith('.nii.gz')):
        raise HTTPException(
            status_code=400,
            detail="Only NIfTI files (.nii or .nii.gz) are supported"
        )

    # 保存临时文件
    ct_path = f"/tmp/{file.filename}"
    try:
        content = await file.read()
        with open(ct_path, "wb") as f:
            f.write(content)

        # 运行完整pipeline
        result = analyze_ct(ct_path, qdrant_client=qdrant)

        # 记录监控数据
        log_result(file.filename, result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(ct_path):
            os.remove(ct_path)

@app.get("/stats")
def get_stats():
    """系统性能统计"""
    conn = sqlite3.connect(DB_PATH)

    # 总体统计
    total = conn.execute(
        "SELECT COUNT(*) FROM analyses"
    ).fetchone()[0]

    if total == 0:
        conn.close()
        return {"message": "No analyses yet", "total": 0}

    avg_latency = conn.execute(
        "SELECT AVG(latency_s) FROM analyses"
    ).fetchone()[0]

    # 置信度分布
    confidence_dist = conn.execute("""
        SELECT confidence, COUNT(*) as count
        FROM analyses GROUP BY confidence
    """).fetchall()

    # 风险等级分布
    risk_dist = conn.execute("""
        SELECT risk_level, COUNT(*) as count
        FROM analyses GROUP BY risk_level
    """).fetchall()

    # 最近5次
    recent = conn.execute("""
        SELECT timestamp, filename, latency_s,
               confidence, risk_level
        FROM analyses ORDER BY id DESC LIMIT 5
    """).fetchall()

    conn.close()

    return {
        "total_analyses": total,
        "avg_latency_seconds": round(avg_latency, 2),
        "confidence_distribution": {r[0]: r[1] for r in confidence_dist},
        "risk_distribution": {r[0]: r[1] for r in risk_dist},
        "recent_analyses": [
            {
                "timestamp": r[0],
                "file": r[1],
                "latency": r[2],
                "confidence": r[3],
                "risk": r[4]
            }
            for r in recent
        ]
    }

@app.get("/health")
def health():
    """系统健康检查"""
    import torch
    return {
        "status": "ok",
        "gpu": {
            "available": torch.cuda.is_available(),
            "device": str(torch.cuda.get_device_name(0))
                      if torch.cuda.is_available() else "CPU",
            "memory_used_mb": round(
                torch.cuda.memory_allocated(0) / 1024**2, 1
            ) if torch.cuda.is_available() else 0
        },
        "models": {
            "vlm": "Med3DVLM-Qwen-2.5-7B",
            "clip": "DCFormer-SigLIP"
        },
        "paper": "IEEE JBHI 2025 - Med3DVLM"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )

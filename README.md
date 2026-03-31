# Medical VLM Report Assistant

> **CT radiology report generation** using Stanford Merlin (Nature 2026) 3D Vision-Language Model + RAG + GPT-4o — with ICD-10 coding, risk stratification, and clinical guardrails.

[![Nature 2026](https://img.shields.io/badge/Nature-2026-blue)](https://doi.org/10.1038/s41586-026-10181-8)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688)](https://fastapi.tiangolo.com)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-OpenAI-412991)](https://openai.com)

---

## Demo

![Demo](docs/demo.png)

---

## Pipeline
```
CT scan (.nii.gz)
      ↓
Stanford Merlin (Nature 2026)
3D CT VLM → 2048-dim embedding
      ↓
RAG Retrieval
Similar case search (Qdrant)
      ↓
GPT-4o Report Generation
Structured clinical report
      ↓
Output: Findings · Impression · ICD-10 · Risk Level
```

---

## Key Results

| Metric | Value |
|--------|-------|
| CT Embedding | **2048-dim** (Merlin 3D VLM) |
| Similar Cases Retrieved | **3** per scan |
| Avg Pipeline Latency | **~13s** (CPU) |
| Report Sections | Findings · Impression · Recommendations · ICD-10 |
| Risk Levels | LOW / MODERATE / HIGH |

---

## Components

### Stanford Merlin (Nature 2026)
- 3D CT Vision-Language Model
- Pretrained on 1.8M CT scans + EHR data
- Extracts 2048-dimensional image embeddings
- [Paper](https://doi.org/10.1038/s41586-026-10181-8) | [HuggingFace](https://huggingface.co/stanfordmimi/Merlin)

### RAG Pipeline
- Qdrant vector database for similar case retrieval
- sentence-transformers embeddings
- 5 curated radiology report templates

### GPT-4o Report Generation
- Structured output: Findings, Impression, Recommendations, ICD-10 codes
- Risk stratification: LOW / MODERATE / HIGH
- Clinical context from retrieved similar cases

### FastAPI + React Frontend
- CT file upload (.nii / .nii.gz)
- Demo mode with Merlin sample CT
- Live metrics dashboard
- Real-time pipeline progress visualization

---

## Quick Start
```bash
# 1. Create environment
conda create -n merlin_env python=3.10
conda activate merlin_env

# 2. Install dependencies
pip install merlin-vlm fastapi uvicorn openai qdrant-client sentence-transformers python-multipart rich pandas requests

# 3. Set environment variables
cp .env.example api/.env
# Add your OPENAI_API_KEY

# 4. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 5. Start API
cd api
python vlm_api.py
# Open http://localhost:8001
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| 3D VLM | Stanford Merlin (Nature 2026) |
| LLM | GPT-4o |
| Vector DB | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| API | FastAPI |
| Frontend | React 18 |
| CT Processing | MONAI, NiBabel |

---

## Author

**Hongcheng Jiang** — PhD ECE, UMKC (GPA: 4.0)

9 publications: CVPR · IEEE JSTARS · IEEE SMC · WACV · Infrared Physics & Technology

[![GitHub](https://img.shields.io/badge/GitHub-jianghongcheng-black)](https://github.com/jianghongcheng)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/hongcheng-jiang-a31860181)

"""
Medical VLM CT Report Generation System
Core: Med3DVLM (IEEE JBHI 2025)
Architecture:
  1. Structured Perception (VQA)
  2. Image RAG (DCFormer-SigLIP)
  3. Report Generation + Uncertainty
  4. Safety Layer (Human-in-the-loop)
"""
import os
import sys
import time
import torch
import numpy as np
import SimpleITK as sitk
from typing import Optional

sys.path.insert(0, '/media/max/b/LLM/Med3DVLM')
sys.path.insert(0, '/media/max/b/LLM/medical-vlm-assistant')

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from src.model.CLIP import DEC_CLIP, DEC_CLIPConfig

# ─────────────────────────────────────────
# 全局模型（只加载一次）
# ─────────────────────────────────────────
_vlm_model = None
_vlm_tokenizer = None
_clip_model = None
_clip_tokenizer = None
_proj_out_num = 256
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VLM_PATH  = '/media/max/b/LLM/medical-vlm-assistant/models/Med3DVLM-Qwen-2.5-7B'
CLIP_PATH = '/media/max/b/LLM/medical-vlm-assistant/models/DCFormer_SigLIP'


def load_vlm():
    global _vlm_model, _vlm_tokenizer, _proj_out_num
    if _vlm_model is None:
        print("[VLM] Loading Med3DVLM-Qwen-2.5-7B...")
        _vlm_tokenizer = AutoTokenizer.from_pretrained(
            VLM_PATH, use_fast=False, trust_remote_code=True
        )
        _vlm_model = AutoModelForCausalLM.from_pretrained(
            VLM_PATH,
            torch_dtype=torch.bfloat16,
            device_map=_device,
            trust_remote_code=True
        )
        if hasattr(_vlm_model.get_model().config, 'proj_out_num'):
            _proj_out_num = _vlm_model.get_model().config.proj_out_num
        print(f"[VLM] Ready. proj_out_num={_proj_out_num}")
    return _vlm_model, _vlm_tokenizer


def load_clip():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        print("[CLIP] Loading DCFormer-SigLIP...")
        AutoConfig.register('dec_clip', DEC_CLIPConfig)
        AutoModel.register(DEC_CLIPConfig, DEC_CLIP)
        _clip_tokenizer = AutoTokenizer.from_pretrained(
            CLIP_PATH, trust_remote_code=True
        )
        _clip_model = AutoModel.from_pretrained(
            CLIP_PATH,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(_device)
        _clip_model.eval()
        print("[CLIP] Ready.")
    return _clip_model, _clip_tokenizer


# ─────────────────────────────────────────
# CT 预处理
# ─────────────────────────────────────────
def load_ct(ct_path: str) -> torch.Tensor:
    """读取NIfTI CT，返回 (1,D,H,W) float32"""
    img = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
    arr = np.expand_dims(img, axis=0).astype(np.float32)
    return arr


# ─────────────────────────────────────────
# 阶段1：结构化感知（VQA）
# 论文Table IV: 器官74.75%, 异常66.65%
# ─────────────────────────────────────────
STRUCTURED_QUESTIONS = {
    "plane":       "Which imaging plane is displayed? (Axial/Sagittal/Coronal)",
    "phase":       "What is the CT phase? (Non-contrast/Arterial/Portal venous/Delayed)",
    "organ":       "Which organ shows the most significant abnormality?",
    "abnormality": "What type of abnormality is present? (Mass/Cyst/Thrombus/Inflammation/Normal)",
    "location":    "Where is the abnormality located? (Left/Right/Upper/Lower/Central)",
    "severity":    "How would you rate the severity? (Mild/Moderate/Severe/Normal)"
}

def structured_perception(ct_array: np.ndarray) -> dict:
    """
    逐问题询问Med3DVLM，得到结构化中间表示
    基于论文VQA能力（79.95% closed-ended accuracy）
    """
    model, tokenizer = load_vlm()
    img_pt = torch.from_numpy(ct_array).unsqueeze(0).to(
        dtype=torch.bfloat16, device=_device
    )
    image_tokens = '<im_patch>' * _proj_out_num
    findings = {}

    for key, question in STRUCTURED_QUESTIONS.items():
        input_txt = image_tokens + question
        input_id = tokenizer(
            input_txt, return_tensors='pt'
        )['input_ids'].to(_device)

        with torch.no_grad():
            gen = model.generate(
                images=img_pt,
                inputs=input_id,
                max_new_tokens=64,
                do_sample=False,   # greedy for structured questions
                temperature=1.0
            )
        answer = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        findings[key] = answer.strip()

    return findings


# ─────────────────────────────────────────
# 阶段2：图像RAG（DCFormer-SigLIP）
# 论文Table I: Recall@1=61.00% (2000候选)
# ─────────────────────────────────────────
def extract_image_embedding(ct_array: np.ndarray) -> np.ndarray:
    """提取CT的图像embedding用于RAG检索"""
    model, _ = load_clip()
    img_pt = torch.from_numpy(ct_array).unsqueeze(0).float().to(_device)
    with torch.no_grad():
        emb = model.encode_image(img_pt)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


def image_rag_search(ct_array: np.ndarray, qdrant_client=None, top_k: int = 3) -> list:
    """
    图像到图像RAG检索
    返回最相似的历史CT案例作为推理参考
    注意：仅作辅助推理，不直接注入报告
    """
    if qdrant_client is None:
        return []  # Qdrant未初始化时跳过RAG

    embedding = extract_image_embedding(ct_array)
    try:
        results = qdrant_client.query_points(
            collection_name="ct_image_rag",
            query=embedding[0].tolist(),
            limit=top_k
        )
        return [
            {
                "score": r.score,
                "case_id": r.payload.get("case_id", "unknown"),
                "finding_summary": r.payload.get("finding_summary", ""),
                "diagnosis": r.payload.get("diagnosis", "")
            }
            for r in results.points
        ]
    except Exception as e:
        print(f"[RAG] Search failed: {e}")
        return []


# ─────────────────────────────────────────
# 阶段3：报告生成 + 不确定性估计
# 论文METEOR=36.42%, 但承认有幻觉问题
# ─────────────────────────────────────────
REPORT_PROMPT_TEMPLATE = """CT report. {organ} {abnormality} {location}. {rag_context}
FINDINGS:"""


def generate_report_with_uncertainty(
    ct_array: np.ndarray,
    structured_findings: dict,
    similar_cases: list,
    n_samples: int = 1
) -> dict:
    """
    生成报告并估计不确定性
    通过多次采样计算一致性作为confidence
    基于论文Future Directions: uncertainty-aware response generation
    """
    model, tokenizer = load_vlm()
    img_pt = torch.from_numpy(ct_array).unsqueeze(0).to(
        dtype=torch.bfloat16, device=_device
    )

    # 构建RAG上下文（辅助推理，不直接注入）
    rag_context = ""
    if similar_cases:
        rag_context = "Similar cases for reasoning reference only (do not copy):\n"
        for i, case in enumerate(similar_cases[:2], 1):
            rag_context += f"Case {i} (similarity={case['score']:.2f}): {case['finding_summary']}\n"

    prompt = REPORT_PROMPT_TEMPLATE.format(
        **structured_findings,
        rag_context=rag_context
    )

    image_tokens = '<im_patch>' * _proj_out_num
    input_txt = image_tokens + prompt
    # 注意：不能truncation，否则图像token会被截掉
    # image_tokens已经是256个token，text prompt必须很短
    input_id = tokenizer(
        input_txt, return_tensors='pt'
    )['input_ids'].to(_device)

    # 多次采样估计不确定性
    reports = []
    for i in range(n_samples):
        with torch.no_grad():
            gen = model.generate(
                images=img_pt,
                inputs=input_id,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7 + i * 0.1  # 略微变化temperature
            )
        report = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        reports.append(report)

    # 计算一致性（简单版：比较主要字段）
    primary_report = reports[0]
    confidence_score = _estimate_confidence(reports, structured_findings)

    return {
        "primary_report": primary_report,
        "confidence": confidence_score,
        "n_samples": n_samples,
        "all_samples": reports
    }


def _estimate_confidence(reports: list, structured_findings: dict) -> str:
    """
    估计报告置信度
    基于：结构化感知的确定性 + 多次采样的一致性
    """
    # 检查structured_findings中的不确定关键词
    uncertain_keywords = ['normal', 'unclear', 'unknown', 'cannot', 'uncertain']
    uncertain_count = sum(
        1 for v in structured_findings.values()
        if any(kw in v.lower() for kw in uncertain_keywords)
    )

    # 检查异常类型
    abnormality = structured_findings.get('abnormality', '').lower()
    if 'normal' in abnormality:
        base_confidence = 'HIGH'
    elif uncertain_count >= 2:
        base_confidence = 'LOW'
    else:
        base_confidence = 'MEDIUM'

    # 多次采样一致性检查（简化版）
    if len(reports) >= 2:
        # 比较前两次报告的RISK_LEVEL
        risk_levels = []
        for r in reports:
            for level in ['HIGH', 'MODERATE', 'LOW']:
                if f'RISK_LEVEL: {level}' in r or f'RISK_LEVEL:{level}' in r:
                    risk_levels.append(level)
                    break

        if len(set(risk_levels)) > 1:
            # 风险等级不一致 → 降低置信度
            if base_confidence == 'HIGH':
                base_confidence = 'MEDIUM'
            elif base_confidence == 'MEDIUM':
                base_confidence = 'LOW'

    return base_confidence


# ─────────────────────────────────────────
# 阶段4：安全兜底
# Human-in-the-loop（永远必须）
# ─────────────────────────────────────────
DISCLAIMER = (
    "⚠️ AI-GENERATED REPORT — FOR RESEARCH USE ONLY. "
    "This report must be reviewed and verified by a qualified radiologist "
    "before any clinical use. Not approved for diagnostic purposes."
)

def apply_safety_layer(report_data: dict, structured_findings: dict) -> dict:
    """
    安全兜底层
    - 永远标记review_required=True
    - 状态永远是DRAFT
    - 添加法律免责声明
    """
    confidence = report_data.get('confidence', 'LOW')
    risk_level = 'UNKNOWN'
    report_text = report_data.get('primary_report', '')

    # 提取RISK_LEVEL
    for level in ['HIGH', 'MODERATE', 'LOW']:
        if level in report_text:
            risk_level = level
            break

    return {
        "status": "DRAFT",
        "review_required": True,
        "human_review_status": "PENDING",
        "disclaimer": DISCLAIMER,
        "report": report_text,
        "structured_findings": structured_findings,
        "confidence": confidence,
        "risk_level": risk_level,
        "safety_flags": _check_safety_flags(report_text, confidence)
    }


def _check_safety_flags(report_text: str, confidence: str) -> list:
    """检查需要特别注意的安全标志"""
    flags = []

    high_risk_terms = [
        'malignancy', 'cancer', 'tumor', 'mass', 'metastasis',
        'hemorrhage', 'thrombosis', 'embolism', 'perforation'
    ]
    for term in high_risk_terms:
        if term in report_text.lower():
            flags.append(f"HIGH_RISK_TERM_DETECTED: {term}")

    if confidence == 'LOW':
        flags.append("LOW_CONFIDENCE: Results may be unreliable")

    if not flags:
        flags.append("ROUTINE_REVIEW_REQUIRED")

    return flags


# ─────────────────────────────────────────
# 主入口：完整pipeline
# ─────────────────────────────────────────
def analyze_ct(ct_path: str, qdrant_client=None) -> dict:
    """
    完整CT分析pipeline
    
    Args:
        ct_path: NIfTI CT文件路径
        qdrant_client: Qdrant客户端（可选，用于图像RAG）
    
    Returns:
        完整的结构化分析结果
    """
    start_total = time.time()
    results = {"pipeline_stages": {}}

    # 读取CT
    ct_array = load_ct(ct_path)
    print(f"[Pipeline] CT loaded: {ct_array.shape}")

    # 阶段1：结构化感知
    print("[Pipeline] Stage 1: Structured Perception...")
    t = time.time()
    structured_findings = structured_perception(ct_array)
    results["pipeline_stages"]["structured_perception"] = {
        "latency_seconds": round(time.time() - t, 2),
        "findings": structured_findings
    }
    print(f"  Findings: {structured_findings}")

    # 阶段2：图像RAG
    print("[Pipeline] Stage 2: Image RAG...")
    t = time.time()
    similar_cases = image_rag_search(ct_array, qdrant_client)
    results["pipeline_stages"]["image_rag"] = {
        "latency_seconds": round(time.time() - t, 2),
        "similar_cases_found": len(similar_cases),
        "cases": similar_cases
    }

    # 阶段3：报告生成+不确定性
    print("[Pipeline] Stage 3: Report Generation...")
    t = time.time()
    report_data = generate_report_with_uncertainty(
        ct_array, structured_findings, similar_cases
    )
    results["pipeline_stages"]["report_generation"] = {
        "latency_seconds": round(time.time() - t, 2),
        "confidence": report_data["confidence"]
    }

    # 阶段4：安全兜底
    print("[Pipeline] Stage 4: Safety Layer...")
    final_result = apply_safety_layer(report_data, structured_findings)

    # 汇总
    total_latency = round(time.time() - start_total, 2)
    final_result.update({
        "pipeline_stages": results["pipeline_stages"],
        "total_latency_seconds": total_latency,
        "model": "Med3DVLM (DCFormer + SigLIP + Qwen2.5-7B)",
        "paper": "IEEE JBHI 2025"
    })

    print(f"[Pipeline] Complete in {total_latency}s")
    return final_result

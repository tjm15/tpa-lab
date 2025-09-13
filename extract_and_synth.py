#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import json
import glob
import shutil
import pathlib
import logging
import subprocess
import threading
import time
import itertools
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# Optional Gemini (only needed for --synth)
_GEMINI_READY = False
try:
    import google.generativeai as genai
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _GEMINI_READY = True
except Exception:
    pass

# ========================= Defaults / Config =========================
# Use an exact local model tag (your 20B MXFP4 build). You can override via OLLAMA_MODEL env.
MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

INDEX_WINDOW_PAGES = int(os.getenv("TPA_INDEX_WINDOW_PAGES", "16"))
GAP_BLOCK_PAGES    = int(os.getenv("TPA_GAP_BLOCK_PAGES", "6"))
INDEX_MAX_CHARS    = int(os.getenv("TPA_INDEX_MAX_CHARS", "20000"))

# Keep worker count modest to avoid head-of-line blocking
DEFAULT_WORKERS = min(4, os.cpu_count() or 4)

# MXFP4-tuned defaults for a 20B model on ~32GB VRAM.
# NOTE: Many runtimes cap context length; override at run-time with TPA_NUM_CTX if needed.
LLM_OPTIONS_DEFAULT = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    # Long default; if your server caps ctx lower, set TPA_NUM_CTX to e.g. 16384 or 32768.
    "num_ctx": int(os.getenv("TPA_NUM_CTX", "196608")),
    "num_thread": os.cpu_count() or 8,
    # MXFP4 + Blackwell likes bigger batches; tune with TPA_NUM_BATCH if you see under-utilisation.
    "num_batch": int(os.getenv("TPA_NUM_BATCH", "2304")),
}

# Prefer streaming generate; flip with env=0 if you want chat API
USE_GENERATE_ONLY = os.getenv("TPA_USE_GENERATE_ONLY", "1") == "1"
CHAT_TIMEOUT = int(os.getenv("TPA_CHAT_TIMEOUT", "0"))  # <=0 => wait forever

# Allow a few in-flight requests; MXFP4 handles this well on 32GB
LLM_MAX_CONCURRENCY = int(os.getenv("TPA_LLM_CONCURRENCY", "3"))
_LLM_SEMA = threading.Semaphore(LLM_MAX_CONCURRENCY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract-policies")

# ========================= Text Normalisation & OCR ====================
def _normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)  # de-hyphenate line breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _pdf_to_pages_pymupdf(pdf_path: str) -> List[str]:
    import fitz
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            try:
                text = doc[i].get_text("text") or ""
            except Exception as e:
                log.debug(f"PyMuPDF failed on page {i+1} of {pdf_path}: {e}")
                text = ""
            pages.append(_normalize_text(text))
    return pages

def _pdf_to_pages_pdfplumber(pdf_path: str) -> List[str]:
    import pdfplumber
    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            try:
                text = p.extract_text() or ""
            except Exception as e:
                log.debug(f"pdfplumber failed on page {i} of {pdf_path}: {e}")
                text = ""
            pages.append(_normalize_text(text))
    return pages

def _pdf_to_pages_pypdf(pdf_path: str) -> List[str]:
    from pypdf import PdfReader
    pages: List[str] = []
    reader = PdfReader(pdf_path, strict=False)
    for i in range(len(reader.pages)):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception as e:
            log.debug(f"pypdf failed on page {i+1} of {pdf_path}: {e}")
            text = ""
        pages.append(_normalize_text(text))
    return pages

def _repair_pdf_with_mutool(src_path: str) -> Optional[str]:
    if shutil.which("mutool") is None:
        return None
    repaired = str(pathlib.Path(src_path).with_suffix(".repaired.pdf"))
    try:
        subprocess.run(["mutool", "clean", "-d", "-l", src_path, repaired],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return repaired if os.path.exists(repaired) else None
    except Exception as e:
        log.debug(f"mutool repair failed for {src_path}: {e}")
        return None

def pdf_to_pages(pdf_path: str) -> List[str]:
    # 1) PyMuPDF
    try:
        pages = _pdf_to_pages_pymupdf(pdf_path)
    except Exception as e:
        log.debug(f"PyMuPDF whole-file failure on {pdf_path}: {e}")
        pages = []
    # 2) pdfplumber
    if not pages or all(len(p.strip()) == 0 for p in pages):
        try:
            pages = _pdf_to_pages_pdfplumber(pdf_path)
        except Exception as e:
            log.debug(f"pdfplumber whole-file failure on {pdf_path}: {e}")
            pages = []
    # 3) pypdf
    if not pages or all(len(p.strip()) == 0 for p in pages):
        try:
            pages = _pdf_to_pages_pypdf(pdf_path)
        except Exception as e:
            log.debug(f"pypdf whole-file failure on {pdf_path}: {e}")
            pages = []
    # 4) repair + retry
    if not pages or all(len(p.strip()) == 0 for p in pages):
        repaired = _repair_pdf_with_mutool(pdf_path)
        if repaired:
            try:
                pages = _pdf_to_pages_pymupdf(repaired)
            except Exception as e:
                log.debug(f"PyMuPDF on repaired failed {repaired}: {e}")
                pages = []
    empties = sum(1 for p in pages if len(p.strip()) == 0)
    if pages and empties:
        log.warning(f"[{pathlib.Path(pdf_path).name}] Empty/failed pages: {empties}/{len(pages)} (skipped but continuing)")
    return pages

def needs_ocr(pages: List[str]) -> bool:
    nonempty = sum(1 for p in pages if len(p.strip()) >= 50)
    return nonempty < max(3, int(0.25 * len(pages)))

def ocr_pdf_if_needed(pdf_path: str) -> Optional[str]:
    if shutil.which("ocrmypdf") is None:
        return None
    tmp_out = str(pathlib.Path(pdf_path).with_suffix(".ocr.pdf"))
    try:
        subprocess.run(["ocrmypdf", "--skip-text", "--force-ocr", pdf_path, tmp_out],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return tmp_out if os.path.exists(tmp_out) else None
    except Exception:
        return None

# ========================= Deterministic pre-index =====================
def _looks_like_title(s: str) -> bool:
    s = s.strip()
    if len(s) < 3 or len(s) > 160:
        return False
    if s.endswith("."):
        return False
    return bool(re.match(r'^[A-Z0-9][A-Za-z0-9 \-/&(),’\'“”]+$', s)) or s.isupper()

POLICY_LINE_PATTERNS = [
    r'^\s*(?:Policy|POLICY)\s+([A-Z]{1,4}\d{0,3}[A-Za-z]?)\s*[:\-–]\s*(.+)$',   # Policy H1: Title
    r'^\s*([A-Z]{1,4}\d{1,3}[A-Za-z]?)\s*[:\-–]\s*(.+)$',                       # H1: Title
    r'^\s*(?:Policy|POLICY)\s+([A-Z]{1,4}\d{0,3}[A-Za-z]?)\s*$',                # Policy H1
    r'^\s*([A-Z][A-Z \-/&]{3,60})\s*$',                                         # ALL-CAPS heading
]

POLICY_KEYWORDS = {
    "policy", "policies", "development will be permitted", "proposals should",
    "requirements", "criteria", "applicants must", "will be supported", "will not be permitted"
}

def parse_toc_candidates(pages: List[str], max_scan_pages: int = 12) -> List[Dict]:
    toc = "\n".join(pages[:min(max_scan_pages, len(pages))])
    if not re.search(r'\b(contents|table of contents|index)\b', toc, re.I):
        return []
    cands = []
    line_re = re.compile(
        r'^\s*(?:Policy\s+)?([A-Z]{1,4}\d{1,3}[A-Za-z]?)\s*[:\-–]\s*([A-Za-z0-9 ,\-/&()’\'“”]+?)\s+(\d{1,4})\s*$',
        re.I
    )
    for p in pages[:max_scan_pages]:
        for ln in p.splitlines():
            m = line_re.match(ln.strip())
            if not m:
                continue
            pid, title, page_no = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
            if 1 <= page_no <= len(pages) and _looks_like_title(title):
                cands.append({
                    "policy_id_guess": pid,
                    "policy_title_guess": title[:160],
                    "page_start": page_no,
                    "page_end": min(len(pages), page_no + 3),
                    "heading_text": f"{pid}: {title}",
                    "is_ambiguous": False,
                    "confidence": "high",
                    "notes": "from ToC"
                })
    return cands

def greedy_heading_preindex(pages: List[str]) -> Dict:
    candidates = []
    for i, page in enumerate(pages, start=1):
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        top = lines[:60]
        j = 0
        while j < len(top):
            ln = top[j]
            for pat in POLICY_LINE_PATTERNS:
                m = re.match(pat, ln)
                if not m:
                    continue
                pid, title = "", ""
                if pat == POLICY_LINE_PATTERNS[0]:
                    pid, title = m.group(1).strip(), m.group(2).strip()
                elif pat == POLICY_LINE_PATTERNS[1]:
                    pid, title = m.group(1).strip(), m.group(2).strip()
                elif pat == POLICY_LINE_PATTERNS[2]:
                    pid = m.group(1).strip()
                    if j + 1 < len(top) and _looks_like_title(top[j+1]):
                        title = top[j+1].strip(); j += 1
                else:
                    maybe_title = m.group(1).strip()
                    prev = top[j-1] if j > 0 else ""
                    prev_id = re.match(r'^\s*(?:Policy\s+)?([A-Z]{1,4}\d{1,3}[A-Za-z]?)\s*$', prev, re.I)
                    if prev_id and _looks_like_title(maybe_title):
                        pid, title = prev_id.group(1).strip(), maybe_title
                    elif any(k in ln.lower() for k in POLICY_KEYWORDS):
                        title = maybe_title
                if title and len(title) > 160:
                    title = title[:160].rstrip()
                if pid or title:
                    bad = re.search(r'^(contents|list of|figures?|tables?)\b', (title or "").lower())
                    if not bad:
                        candidates.append({
                            "page": i,
                            "policy_id_guess": pid,
                            "policy_title_guess": title,
                            "heading_text": ln if title != ln else f"{pid} {title}".strip()
                        })
                        break
            j += 1

    policies = []
    for a, b in itertools.zip_longest(candidates, candidates[1:]):
        start = a["page"]
        natural_end = (b["page"] - 1) if b and b["page"] >= start else start
        end = min(natural_end, start + 7)
        title = (a.get("policy_title_guess") or "").strip()
        pid   = (a.get("policy_id_guess") or "").strip()
        if not pid and not _looks_like_title(title):
            continue
        policies.append({
            "policy_id_guess": pid,
            "policy_title_guess": title,
            "page_start": start,
            "page_end": max(start, end),
            "heading_text": a["heading_text"],
            "is_ambiguous": not bool(pid and title),
            "confidence": "medium" if (pid or title) else "low",
            "notes": "deterministic heading match"
        })
    coverage = [{"from_page": 1, "to_page": len(pages), "label": "non_policy"}]
    for p in policies:
        coverage.append({"from_page": p["page_start"], "to_page": p["page_end"], "label": "policy"})
    return {"policies": policies, "coverage": merge_coverage_segments(coverage)}

# ========================= LLM (Ollama) wiring =========================
def list_local_models(base_url: str) -> set[str]:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        tags = r.json().get("models", [])
        return {m.get("name") or m.get("model") for m in tags}
    except Exception:
        return set()

def pull_model_exact(base_url: str, model_tag: str) -> None:
    resp = requests.post(f"{base_url.rstrip('/')}/api/pull",
                         json={"model": model_tag}, timeout=3600, stream=True)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            err = e.response.json()
        except Exception:
            err = e.response.text
        raise SystemExit(f"Failed to pull '{model_tag}': {err}") from e
    for _ in resp.iter_lines():
        pass

def resolve_model_or_exit(base_url: str, wanted: str, allow_autopull: bool) -> str:
    if wanted.endswith(":latest"):
        raise SystemExit("Refusing ':latest'. Specify an exact tag (e.g. 'qwen2.5:7b-instruct-q5_K_M').")
    local = list_local_models(base_url)
    if wanted in local:
        return wanted
    if not allow_autopull:
        have = ", ".join(sorted(local)) if local else "none"
        raise SystemExit(f"Model '{wanted}' not found. Local: {have}. Pull it or re-run with --auto-pull.")
    print(f"[model] Auto-pulling exact tag '{wanted}' ...")
    pull_model_exact(base_url, wanted)
    if wanted not in list_local_models(base_url):
        raise SystemExit(f"Pull finished but '{wanted}' not listed. Aborting.")
    return wanted

def resolve_ollama_url() -> str:
    env = os.getenv("OLLAMA_URL")
    if env:
        return env.rstrip("/")
    candidates = []
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    ip = line.split()[1].strip()
                    candidates.append(f"http://{ip}:11434")
                    break
    except Exception:
        pass
    candidates += ["http://127.0.0.1:11434", "http://localhost:11434"]
    for base in candidates:
        try:
            r = requests.get(f"{base}/api/version", timeout=1.5)
            if r.ok:
                return base
        except Exception:
            continue
    return "http://127.0.0.1:11434"

def _stream_generate(base: str, model: str, prompt: str, opts: Dict, timeout: int) -> str:
    to = None if timeout <= 0 else timeout
    with _LLM_SEMA:
        t0 = time.time()
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True, "options": opts},
            timeout=to, stream=True,
        )
        r.raise_for_status()
        out_parts: List[str] = []
        first_at: Optional[float] = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line if isinstance(line, str) else line.decode("utf-8", "ignore"))
            except Exception:
                continue
            if "response" in obj:
                if first_at is None:
                    first_at = time.time()
                    log.info(f"[llm] first token in {first_at - t0:.2f}s")
                out_parts.append(obj["response"])
            if obj.get("done"):
                break
        if first_at is not None:
            t1 = time.time()
            text = "".join(out_parts)
            toks = max(1, len(text) // 4)
            dt = max(1e-2, t1 - first_at)
            log.info(f"[llm] ~{toks} toks in {dt:.2f}s (~{toks/dt:.1f} tok/s)")
    return "".join(out_parts).strip()

def chat_ollama(messages: List[Dict], model: Optional[str] = None,
                base_url: Optional[str] = None, options: Optional[Dict] = None,
                timeout: int = CHAT_TIMEOUT, json_mode: bool = True) -> str:
    model = model or MODEL_DEFAULT
    base = (base_url or resolve_ollama_url()).rstrip("/")
    opts = dict(LLM_OPTIONS_DEFAULT)
    if options:
        opts.update(options)
    if json_mode:
        # enables JSON grammar in many Ollama models
        opts["format"] = "json"

    system_text = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    user_text   = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
    prompt = (system_text + "\n\n" + user_text).strip() if system_text else user_text

    if USE_GENERATE_ONLY:
        return _stream_generate(base, model, prompt, opts, timeout)

    ep = f"{base}/api/chat"
    payload = {"model": model, "messages": messages, "options": opts, "stream": False}
    with _LLM_SEMA:
        r = requests.post(ep, json=payload, timeout=None if timeout <= 0 else timeout)

    ctype = r.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            return (msg.get("content") or "").strip()
        choices = data.get("choices") or data.get("messages") or []
        if isinstance(choices, list) and choices:
            maybe = choices[-1]
            if isinstance(maybe, dict):
                return (maybe.get("message", {}).get("content") or maybe.get("content") or "").strip()
        return (r.text or "").strip()
    return _stream_generate(base, model, prompt, opts, timeout)

# ========================= Robust JSON parsing =========================
def to_json(s: str, context_label: str = "response") -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    best = None; stack = 0; start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start is not None:
                candidate = s[start:i+1]
                if best is None or len(candidate) > len(best):
                    best = candidate
                start = None
    if best:
        try:
            return json.loads(best)
        except Exception:
            pass
    raise ValueError(f"Failed to parse JSON ({context_label}). len={len(s)} head={s[:160]!r} tail={s[-160:]!r}")

# ========================= Prompts =====================================
SYSTEM_RECALL = (
    "You are a meticulous UK planning officer. Maximise recall of policy sections.\n"
    "Err on inclusion: if a section is plausibly a policy, include it with is_ambiguous:true and explain in notes.\n"
    "Do not invent beyond the provided text. Output JSON only when asked. No Markdown."
)

SYSTEM_STRICT = (
    "You are a meticulous UK planning officer.\n"
    "Rules:\n"
    "- ONLY copy facts explicitly present in the provided TEXT span.\n"
    "- If a field is not present, OMIT it; do NOT guess.\n"
    "- Any numeric target MUST include an 'evidence_quotes' item (≤25 words) with page number.\n"
    "- If the span clearly does NOT contain a policy, return: {\"skip\": true, \"reason\": \"no policy content in span\"}.\n"
    "- Output JSON ONLY. No prose, no Markdown."
)

def build_recall_index_prompt(pages: List[str]) -> str:
    doc_text = [f"<<<PAGE {i}>>>\n{t}" for i, t in enumerate(pages, start=1)]
    joined = "\n".join(doc_text)
    while len(joined) > INDEX_MAX_CHARS and len(doc_text) > 4:
        doc_text.pop()
        joined = "\n".join(doc_text)
    return (
        "TASK: Build a HIGH-RECALL index of *policies* in this document.\n\n"
        "RETURN JSON ONLY with:\n"
        "{\n"
        "  \"policies\": [ { \"policy_id_guess\":\"H1\",\"policy_title_guess\":\"Housing Mix\",\"page_start\":12,\"page_end\":15,\"heading_text\":\"Policy H1: Housing Mix\",\"alt_headings\":[\"H1 Housing Mix\",\"Policy H1\"],\"is_ambiguous\":false,\"confidence\":\"high\",\"notes\":\"Why included or uncertain\" } ],\n"
        "  \"coverage\":[ {\"from_page\":1,\"to_page\":11,\"label\":\"non_policy\"} ],\n"
        "  \"unassigned_headings\":[ {\"page\":33,\"text\":\"...\"} ]\n"
        "}\n\n"
        "GUIDANCE\n- Include any section that could plausibly be a policy; if unsure set is_ambiguous:true and explain in notes.\n"
        "- Prefer page ranges; char offsets not required.\n"
        "- Over-inclusion is acceptable; validation will prune later.\n\n"
        f"DOCUMENT:\n{joined}"
    )

# ========================= Index & Sweep (optional LLM) ================
def merge_coverage_segments(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (int(s["from_page"]), int(s["to_page"])))
    merged = [segs[0]]
    for s in segs[1:]:
        last = merged[-1]
        if s["label"] == last["label"] and int(s["from_page"]) <= int(last["to_page"]) + 1:
            last["to_page"] = max(int(last["to_page"]), int(s["to_page"]))
        else:
            merged.append(s)
    return merged

def index_policies_recall(pages: List[str], model: str, base_url: str, json_mode=True) -> Dict:
    resp = chat_ollama(
        [{"role": "system", "content": SYSTEM_RECALL},
         {"role": "user", "content": build_recall_index_prompt(pages)}],
        model=model, base_url=base_url, json_mode=json_mode
    )
    data = to_json(resp, context_label="index_policies_recall")
    data.setdefault("policies", [])
    data.setdefault("coverage", [])
    data.setdefault("unassigned_headings", [])
    return data

def gap_blocks(coverage: List[Dict], total_pages: int, block_size: int = GAP_BLOCK_PAGES) -> List[Dict]:
    labels = ["non_policy"] * (total_pages + 1)
    for seg in coverage:
        if seg.get("label") == "policy":
            for p in range(max(1), min(total_pages, int(seg["to_page"])) + 1):
                from_p = max(1, int(seg["from_page"]))
                if from_p <= p <= int(seg["to_page"]):
                    labels[p] = "policy"
    blocks = []
    start = None
    for p in range(1, total_pages + 1):
        if labels[p] == "non_policy" and start is None:
            start = p
        if (labels[p] == "policy" or p == total_pages) and start is not None:
            end = p - 1 if labels[p] == "policy" else p
            i = start
            while i <= end:
                j = min(end, i + block_size - 1)
                blocks.append({"from_page": i, "to_page": j})
                i = j + 1
            start = None
    return blocks

def sweep_gap_for_policies(pages: List[str], start_p: int, end_p: int, model: str, base_url: str, json_mode=True) -> List[Dict]:
    span = [f"<<<PAGE {i}>>>\n{pages[i-1]}" for i in range(start_p, end_p + 1)]
    prompt = (
        f"TASK: GAP SWEEP for missed policies (HIGH RECALL).\n\n"
        f"INPUT: pages {start_p}–{end_p}.\n"
        "Return JSON ONLY:\n"
        "{\n  \"policies\": [ /* same item schema as main index */ ],\n  \"found_any\": true|false }\n\n"
        "GUIDANCE\n- Err on inclusion; set is_ambiguous:true and add notes.\n- If none, set found_any:false and policies:[].\n\n"
        f"TEXT:\n{'\n'.join(span)}"
    )
    try:
        resp = chat_ollama(
            [{"role": "system", "content": SYSTEM_RECALL},
             {"role": "user", "content": prompt}],
            model=model, base_url=base_url, json_mode=json_mode
        )
        data = to_json(resp, context_label=f"gap_sweep {start_p}-{end_p}")
        return data.get("policies", []) if isinstance(data, dict) else []
    except Exception as e:
        if end_p - start_p + 1 <= 2:
            log.error(f"[gap] giving up on tiny slice {start_p}-{end_p}: {e}")
            return []
        mid = (start_p + end_p) // 2
        left = sweep_gap_for_policies(pages, start_p, mid, model, base_url, json_mode=json_mode)
        right = sweep_gap_for_policies(pages, mid + 1, end_p, model, base_url, json_mode=json_mode)
        return left + right

# ========================= Extraction ==================================
def slice_pages(pages: List[str], pstart: int, pend: int) -> str:
    pstart = max(1, pstart)
    pend = min(len(pages), pend)
    return "\n".join([f"<<<PAGE {i}>>>\n{pages[i-1]}" for i in range(pstart, pend + 1)])

def _is_meaningful_page(txt: str) -> bool:
    return len((txt or "").strip()) >= 150

def _tighten_span(pages: List[str], pstart: int, pend: int) -> Tuple[int, int]:
    pstart = max(1, pstart)
    pend = min(len(pages), pend)
    while pstart < pend and not _is_meaningful_page(pages[pstart - 1]):
        pstart += 1
    while pend > pstart and not _is_meaningful_page(pages[pend - 1]):
        pend -= 1
    return pstart, pend

def extract_policy(doc_id: str, pages: List[str], pol: Dict, model: str, base_url: str) -> Dict:
    pstart, pend = int(pol["page_start"]), int(pol["page_end"])
    span_text = slice_pages(pages, pstart, pend)
    user = (
        f"TASK: Extract one policy into strict JSON.\n\n"
        f"CONTEXT\ndoc_id: {doc_id}\nUse only the text from pages {pstart} to {pend}. "
        "If numeric targets or cross-references are present, include verbatim quotes with page numbers.\n\n"
        "SCHEMA\nReturn JSON ONLY matching:\n"
        "{\n"
        "  \"doc_id\":\"string\",\n"
        "  \"policy_id\":\"string\",\n"
        "  \"policy_title\":\"string\",\n"
        "  \"section_label\":\"string|null\",\n"
        f"  \"page_start\": {pstart},\n"
        f"  \"page_end\": {pend},\n"
        "  \"objectives\":[\"string\"],\n"
        "  \"requirements\":[{\"req_id\":\"R1\",\"text\":\"string\",\"type\":\"mandatory|advisory\"}],\n"
        "  \"numeric_targets\":[{\"metric\":\"string\",\"operator\":\"<=|<|=|>=|>\",\"value\": number,\"unit\":\"string\",\"applies_to\":\"string\"}],\n"
        "  \"cross_references\":[\"string\"],\n"
        "  \"geographic_mentions\":[\"string\"],\n"
        "  \"evidence_quotes\":[{\"page\": number, \"quote\": \"string\"}],\n"
        "  \"confidence\":\"high|medium|low\"\n"
        "}\n\n"
        "RULES\n- Do not invent policy IDs or titles; if absent, set a concise best-guess and mark confidence 'low'.\n"
        "- Every numeric target must have a supporting evidence_quotes item.\n"
        "- Keep quotes ≤ 25 words, verbatim.\n"
        "- If something isn’t present, omit it (don’t hallucinate).\n\n"
        f"TEXT (pages {pstart}–{pend}):\n{span_text}"
    )
    resp = chat_ollama(
        [{"role": "system", "content": SYSTEM_STRICT},
         {"role": "user", "content": user}],
        model=model, base_url=base_url, json_mode=True
    )
    obj = to_json(resp, context_label=f"extract_policy p{pstart}-{pend}")

    if obj.get("skip") is True:
        obj.setdefault("validation_notes", []).append(obj.get("reason", "skip"))
        obj.setdefault("confidence", "low")
        obj.setdefault("doc_id", doc_id)
        return obj

    if obj.get("page_start") != pstart or obj.get("page_end") != pend:
        obj.setdefault("validation_notes", []).append("Page span mismatch with requested range.")
    obj.setdefault("doc_id", doc_id)
    obj.setdefault("policy_id", (pol.get("policy_id_guess", "") or ""))
    obj.setdefault("policy_title", (pol.get("policy_title_guess", "") or ""))

    # Ground quotes: keep only verbatim matches in span
    span_lower = span_text.lower()
    good_quotes = []
    for q in obj.get("evidence_quotes", []) or []:
        qt = (q.get("quote") or "").strip()
        if qt and qt.lower() in span_lower:
            good_quotes.append(q)
    obj["evidence_quotes"] = good_quotes
    if not obj["evidence_quotes"]:
        obj.setdefault("validation_notes", []).append("No verifiable quotes in span")
        obj["confidence"] = "low"

    return obj

# ========================= Streaming JSONL write =======================
def append_jsonl(out_path: str, obj: dict) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log.info(f"[write] -> {out_path}")

# ========================= Per-PDF pipeline ============================
def process_pdf(pdf_path: str, out_dir: str, model: str, base_url: str,
                show_inner_pb: bool, preview_chars: int, dry_run: bool,
                use_llm_recall: bool = False) -> Tuple[str, int, int]:
    """
    Returns (out_path, ok_count, total_count)
    """
    doc_id = pathlib.Path(pdf_path).stem

    # Parse text (+ OCR fallback)
    raw_pages = pdf_to_pages(pdf_path)
    if needs_ocr(raw_pages):
        maybe = ocr_pdf_if_needed(pdf_path)
        pages = pdf_to_pages(maybe) if maybe else raw_pages
        if maybe:
            log.info(f"[ocr] Performed OCR for {pathlib.Path(pdf_path).name}")
    else:
        pages = raw_pages
    if not any(pages) or all(len(p.strip()) < 50 for p in pages):
        raise RuntimeError("No extractable text after OCR. Skipping.")

    # 1) ToC seed + greedy regex
    toc_seed = parse_toc_candidates(pages)
    pre = greedy_heading_preindex(pages)
    policies = toc_seed + pre.get("policies", [])
    coverage = merge_coverage_segments((pre.get("coverage", []) or []) + [
        {"from_page": p["page_start"], "to_page": p["page_end"], "label": "policy"} for p in toc_seed
    ])

    # 1b) Optional LLM high-recall top-up (kept small, structured)
    if use_llm_recall:
        idx = index_policies_recall(pages, model, base_url, json_mode=True)
        policies.extend(idx.get("policies", []))
        coverage = merge_coverage_segments(coverage + idx.get("coverage", []))
        blocks = gap_blocks(coverage, total_pages=len(pages), block_size=GAP_BLOCK_PAGES)
        for blk in blocks:
            found = sweep_gap_for_policies(pages, blk["from_page"], blk["to_page"], model, base_url, json_mode=True)
            for pol in found:
                pol.setdefault("is_ambiguous", True)
                pol.setdefault("confidence", "low")
                pol.setdefault("notes", "Added by gap sweep (recall mode)")
            policies.extend(found)

    # de-dup
    dedup = {}
    for pol in policies:
        key = (int(pol.get("page_start", -1)), int(pol.get("page_end", -1)), (pol.get("heading_text") or "").strip())
        if key not in dedup:
            dedup[key] = pol
        else:
            a, b = dedup[key], pol
            if (b.get("page_end", 0) - b.get("page_start", 0)) > (a.get("page_end", 0) - a.get("page_start", 0)):
                dedup[key] = b
    policies = list(dedup.values())

    out_path = os.path.join(out_dir, f"{doc_id}.jsonl")
    ok = 0
    total = 0

    iterator = enumerate(policies, start=1)
    if show_inner_pb and len(policies) > 0:
        iterator = tqdm(iterator, total=len(policies), desc=f"{doc_id} policies", leave=False)

    for _, pol in iterator:
        total += 1
        try:
            # tighten + one extension try if needed
            pstart, pend = _tighten_span(pages, int(pol["page_start"]), int(pol["page_end"]))
            pol["page_start"], pol["page_end"] = pstart, pend
            obj = extract_policy(doc_id, pages, pol, model, base_url)
            if (obj.get("skip") is True or obj.get("confidence") == "low") and pend < len(pages):
                pstart2, pend2 = _tighten_span(pages, pstart, min(len(pages), pend + 1))
                if (pstart2, pend2) != (pstart, pend):
                    pol["page_start"], pol["page_end"] = pstart2, pend2
                    obj = extract_policy(doc_id, pages, pol, model, base_url)

            if preview_chars > 0:
                head = obj.get("policy_title") or obj.get("policy_id") or ""
                sample = json.dumps({
                    "reqs": obj.get("requirements", [])[:2],
                    "nums": obj.get("numeric_targets", [])[:2],
                    "quotes": obj.get("evidence_quotes", [])[:1],
                }, ensure_ascii=False)[:preview_chars]
                log.info(f"[preview] {doc_id} p{pol.get('page_start')}-{pol.get('page_end')}: {head} | {sample}…")

            if not dry_run:
                append_jsonl(out_path, obj)
            ok += 1
        except Exception as e:
            log.exception(f"[ERROR] extract {doc_id} {pol.get('page_start')}-{pol.get('page_end')}: {e}")
            if not dry_run:
                append_jsonl(out_path, {"doc_id": doc_id, "error": str(e), "policy_span": pol})
    return out_path, ok, total

# ========================= Synthesis (Gemini 2.5) ======================
SYNTHESIS_PROMPT = """You are a UK planning officer. Draft a structured report section based ONLY on the provided structured data and quotes.

Rules:
- No facts beyond sources. If unsure, say so explicitly.
- Every numeric or decisive claim MUST have an inline citation like (Doc:{doc_id} p.{page}).
- Keep tone professional and concise. Avoid policy recitation; focus on material considerations and planning balance.

Return Markdown with these headings:
1. Summary (2–3 sentences)
2. Relevant Policies (bullet list with brief relevance + citations)
3. Assessment (clear reasoning; cite quotes near claims)
4. Risks & Uncertainties (list)
5. Recommendation (approve/refuse/conditions) with 1–2 reasons
"""

def build_report_bundle(doc_id: str, jsonl_path: str) -> dict:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("error"):
                continue
            items.append(item)

    policies = [x for x in items if not x.get("skip")]
    quotes = []
    for p in policies:
        for q in (p.get("evidence_quotes") or []):
            if q.get("quote"):
                quotes.append({"doc_id": doc_id, "page": q.get("page"), "quote": q.get("quote", "")[:160]})
    return {"doc_id": doc_id, "policies": policies, "quotes": quotes}

def _require_gemini():
    if not _GEMINI_READY:
        raise SystemExit("Gemini deps missing. Install: pip install google-generativeai tenacity")

def _configure_gemini():
    _require_gemini()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY for --synth.")
    genai.configure(api_key=api_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
       retry=retry_if_exception_type(Exception))
def gemini_synthesise(report_inputs, model_id: str, temp: float = 0.2, max_tokens: int = 2200) -> str:
    _configure_gemini()
    content = [
        {"role": "user", "parts": [
            {"text": SYNTHESIS_PROMPT},
            {"text": "SITE CONTEXT:\n" + (report_inputs.get("site_context") or "N/A")},
            {"text": "POLICIES (structured JSON):\n" + json.dumps(report_inputs.get("policies", [])[:60], ensure_ascii=False)},
            {"text": "EVIDENCE QUOTES (verbatim with pages):\n" + json.dumps(report_inputs.get("quotes", [])[:200], ensure_ascii=False)},
            {"text": "Draft the report now, following the rules."}
        ]}
    ]
    model = genai.GenerativeModel(model_id)
    resp = model.generate_content(
        content,
        generation_config={
            "temperature": temp,
            "top_p": 0.9,
            "max_output_tokens": max_tokens
        }
    )
    return (resp.text or "").strip()

def synth_for_jsonl(jsonl_path: str, out_dir: str, site_context: str, tier: str = "draft", explicit_model: Optional[str] = None) -> str:
    doc_id = pathlib.Path(jsonl_path).stem
    bundle = build_report_bundle(doc_id, jsonl_path)
    bundle["site_context"] = site_context or "N/A"

    model_default = {
        "draft": "models/gemini-2.5-flash",
        "pro":   "models/gemini-2.5-pro"
    }.get(tier, "models/gemini-2.5-flash")
    model_id = explicit_model or os.getenv("GEMINI_MODEL", model_default)

    md = gemini_synthesise(bundle, model_id=model_id, temp=0.2, max_tokens=2200)
    out_md = os.path.join(out_dir, f"{doc_id}.report.{tier}.md")
    pathlib.Path(out_md).write_text(md, encoding="utf-8")
    log.info(f"[synth] -> {out_md} ({model_id})")
    return out_md

# ========================= CLI helpers =================================
def iter_pdf_paths(input_path: str) -> List[str]:
    p = pathlib.Path(input_path)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [str(p.resolve())]
    if p.is_dir():
        return [str(x.resolve()) for x in p.rglob("*.pdf")]
    return [str(pathlib.Path(x).resolve()) for x in glob.glob(input_path)]

def parse_flag_int(argv: List[str], flag: str, default: int) -> int:
    if flag in argv:
        try:
            return int(argv[argv.index(flag) + 1])
        except Exception:
            pass
    return default

def parse_flag_str(argv: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    if flag in argv:
        try:
            return argv[argv.index(flag) + 1]
        except Exception:
            pass
    return default

def has_flag(argv: List[str], flag: str) -> bool:
    return flag in argv

# ========================= main =======================================
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python extract_policies.py <input.pdf|folder|glob> <out_dir> "
              "[--workers N] [--auto-pull] [--preview N] [--dry-run] "
              "[--index-window-pages N] [--gap-block-pages N] "
              "[--llm-recall] "
              "[--synth draft|pro] [--site-context \"text\"] [--gemini-model models/gemini-2.5-flash]")
        sys.exit(1)

    input_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"[init] Output dir: {pathlib.Path(out_dir).resolve()}")

    workers = parse_flag_int(sys.argv, "--workers", DEFAULT_WORKERS)
    preview_chars = parse_flag_int(sys.argv, "--preview", 0)
    dry_run = has_flag(sys.argv, "--dry-run")
    allow_autopull = has_flag(sys.argv, "--auto-pull")
    use_llm_recall = has_flag(sys.argv, "--llm-recall")

    # Optional CLI overrides for window sizes
    global INDEX_WINDOW_PAGES, GAP_BLOCK_PAGES
    INDEX_WINDOW_PAGES = parse_flag_int(sys.argv, "--index-window-pages", INDEX_WINDOW_PAGES)
    GAP_BLOCK_PAGES    = parse_flag_int(sys.argv, "--gap-block-pages", GAP_BLOCK_PAGES)

    # Synthesis controls
    synth_tier = parse_flag_str(sys.argv, "--synth", None)  # None|draft|pro
    site_context = parse_flag_str(sys.argv, "--site-context", "")
    gemini_model_override = parse_flag_str(sys.argv, "--gemini-model", None)

    # Model + endpoint
    wanted_model = MODEL_DEFAULT
    base_url = resolve_ollama_url()
    model = resolve_model_or_exit(base_url, wanted_model, allow_autopull)
    log.info(f"Using local model: {model} @ {base_url}")
    log.info(f"Settings: windows={INDEX_WINDOW_PAGES} pages, max_chars={INDEX_MAX_CHARS}, gap_block={GAP_BLOCK_PAGES}, timeout={CHAT_TIMEOUT}s, generate_only={USE_GENERATE_ONLY}, llm_concurrency={LLM_MAX_CONCURRENCY}")

    pdfs = iter_pdf_paths(input_path)
    if not pdfs:
        log.error("No PDFs found.")
        sys.exit(2)
    log.info(f"Found {len(pdfs)} PDF(s). Using {workers} worker(s). Dry-run={dry_run} Preview={preview_chars} LLM-recall={use_llm_recall}")

    results = []
    errors = 0

    # Extraction
    with tqdm(total=len(pdfs), desc="PDFs", leave=True) as overall_pb:
        if workers <= 1 or len(pdfs) == 1:
            for pdf in pdfs:
                try:
                    out_path, ok, total = process_pdf(
                        pdf, out_dir, model, base_url,
                        show_inner_pb=True, preview_chars=preview_chars, dry_run=dry_run,
                        use_llm_recall=use_llm_recall
                    )
                    results.append((pdf, out_path, ok, total))
                except Exception as e:
                    errors += 1
                    log.exception(f"[ERROR] {pdf}: {e}")
                finally:
                    overall_pb.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        process_pdf, pdf, out_dir, model, base_url,
                        False, preview_chars, dry_run, use_llm_recall
                    ): pdf for pdf in pdfs
                }
                for fut in as_completed(futs):
                    pdf = futs[fut]
                    try:
                        out_path, ok, total = fut.result()
                        results.append((pdf, out_path, ok, total))
                    except Exception as e:
                        errors += 1
                        log.exception(f"[ERROR] {pdf}: {e}")
                    finally:
                        overall_pb.update(1)

    ok_total = sum(ok for _, _, ok, _ in results)
    pol_total = sum(tt for _, _, _, tt in results)
    log.info(f"[summary] Policies extracted: {ok_total}/{pol_total}. Failures: {errors}. Outputs in: {pathlib.Path(out_dir).resolve()}")

    # Synthesis per JSONL (if requested)
    if synth_tier:
        if not _GEMINI_READY:
            raise SystemExit("Install Gemini deps for synthesis: pip install google-generativeai tenacity")
        for pdf, jsonl_path, _, _ in results:
            try:
                _, name = os.path.split(jsonl_path)
                log.info(f"[synth] Building report for {name} ({synth_tier})")
                synth_for_jsonl(jsonl_path, out_dir, site_context, tier=synth_tier, explicit_model=gemini_model_override)
            except Exception as e:
                log.exception(f"[synth ERROR] {jsonl_path}: {e}")

if __name__ == "__main__":
    main()

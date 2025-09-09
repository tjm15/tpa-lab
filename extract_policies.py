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
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ========================= Defaults / Config =========================
MODEL_DEFAULT = "gpt-oss:20b"   # exact tag only (no :latest)

INDEX_WINDOW_PAGES = int(os.getenv("TPA_INDEX_WINDOW_PAGES", "16"))
GAP_BLOCK_PAGES    = int(os.getenv("TPA_GAP_BLOCK_PAGES", "6"))
INDEX_MAX_CHARS    = int(os.getenv("TPA_INDEX_MAX_CHARS", "20000"))

# Keep worker count modest to avoid massive queues/head-of-line blocking.
DEFAULT_WORKERS = min(4, os.cpu_count() or 4)

LLM_OPTIONS_DEFAULT = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.05,
    # Lower default context; 64k is very slow locally unless truly needed.
    "num_ctx": int(os.getenv("TPA_NUM_CTX", "8192")),
    # Speed up CPU tokenisation
    "num_thread": os.cpu_count() or 8,
    # Batch to better saturate a big GPU; adjust via env.
    "num_batch": int(os.getenv("TPA_NUM_BATCH", "1024")),
}

# Prefer generate-only (streaming) by default; flip with env=0 if you want chat
USE_GENERATE_ONLY = os.getenv("TPA_USE_GENERATE_ONLY", "1") == "1"
CHAT_TIMEOUT = int(os.getenv("TPA_CHAT_TIMEOUT", "0"))  # 0 or <0 => no timeout (wait forever)

# Allow some parallelism through to the GPU, but keep it tame by default.
LLM_MAX_CONCURRENCY = int(os.getenv("TPA_LLM_CONCURRENCY", "2"))
_LLM_SEMA = threading.Semaphore(LLM_MAX_CONCURRENCY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract-policies")

# ========================= Safe model resolver =========================
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
        raise SystemExit("Refusing to use ':latest' (may be huge). Specify an exact tag, e.g. 'gpt-oss:20b'.")
    local = list_local_models(base_url)
    if wanted in local:
        return wanted
    if not allow_autopull:
        have = ", ".join(sorted(local)) if local else "none"
        raise SystemExit(
            f"Model '{wanted}' not found.\n"
            f"Locally available: {have}\n"
            f"Pull it: `ollama pull {wanted}` or re-run with --auto-pull."
        )
    print(f"[model] Auto-pulling exact tag '{wanted}' ...")
    pull_model_exact(base_url, wanted)
    local = list_local_models(base_url)
    if wanted not in local:
        raise SystemExit(f"Pull finished but '{wanted}' not listed. Aborting.")
    return wanted

# ========================= Ollama URL auto-detect ======================
def resolve_ollama_url() -> str:
    env = os.getenv("OLLAMA_URL")
    if env:
        return env.rstrip("/")
    candidates = []
    # WSL host IP (Windows host from WSL)
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

# ========================= HTTP → Ollama (streaming) ===================
def _stream_generate(base: str, model: str, prompt: str, opts: Dict, timeout: int) -> str:
    # timeout=None -> wait forever; here 0 or <0 => None
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
            # crude chars→tokens estimate
            toks = max(1, len(text) // 4)
            dt = max(1e-2, t1 - first_at)
            log.info(f"[llm] ~{toks} toks in {dt:.2f}s (~{toks/dt:.1f} tok/s)")
    return "".join(out_parts).strip()

def chat_ollama(
    messages: List[Dict],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    options: Optional[Dict] = None,
    timeout: int = CHAT_TIMEOUT,
    json_mode: bool = True,
) -> str:
    """
    Streaming /api/generate by default (stable locally).
    Keeps /api/chat fallback if you deliberately set TPA_USE_GENERATE_ONLY=0.
    """
    model = model or os.getenv("OLLAMA_MODEL", MODEL_DEFAULT)
    base = (base_url or resolve_ollama_url()).rstrip("/")
    opts = dict(LLM_OPTIONS_DEFAULT)
    if options:
        opts.update(options)
    if json_mode:
        # Note: enabling format=json may invoke a grammar that reduces throughput.
        opts["format"] = "json"

    # Merge messages to a single prompt (system first)
    system_text = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    user_text   = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
    prompt = (system_text + "\n\n" + user_text).strip() if system_text else user_text

    if USE_GENERATE_ONLY:
        return _stream_generate(base, model, prompt, opts, timeout)

    # Optional /api/chat path (not recommended locally)
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
                return (maybe.get("message", {}).get("content")
                        or maybe.get("content")
                        or "").strip()
        return (r.text or "").strip()

    # Fallback to streaming generate if chat response is odd
    return _stream_generate(base, model, prompt, opts, timeout)

# ========================= Robust JSON parsing =========================
def to_json(s: str, context_label: str = "response") -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    # largest balanced {...}
    best = None
    stack = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start is not None:
                candidate = s[start : i + 1]
                if best is None or len(candidate) > len(best):
                    best = candidate
                start = None
    if best:
        try:
            return json.loads(best)
        except Exception:
            pass
    raise ValueError(
        f"Failed to parse JSON ({context_label}). Length={len(s)}; head={s[:160]!r} tail={s[-160:]!r}"
    )

# ========================= PDF parsing (robust) ========================
def _normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _pdf_to_pages_pymupdf(pdf_path: str) -> List[str]:
    import fitz  # pymupdf
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
    pages: List[str] = []
    import pdfplumber
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
    pages: List[str] = []
    from pypdf import PdfReader
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
        subprocess.run(
            ["mutool", "clean", "-d", "-l", src_path, repaired],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
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

# ========================= Prompts =====================================
SYSTEM_RECALL = (
    "You are a meticulous UK planning officer. Maximise recall of policy sections.\n"
    "Err on inclusion: if a section is plausibly a policy, include it with is_ambiguous:true and explain in notes.\n"
    "Do not invent beyond the provided text. Output JSON only when asked. No Markdown."
)

SYSTEM_STRICT = (
    "You are a meticulous UK planning officer. Do not invent content.\n"
    "Work strictly within the provided text span. If unsure, say so.\n"
    "For any numeric target or cross-reference, include a short verbatim quote with page number.\n"
    "Output JSON only when asked. No Markdown."
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
        "  \"policies\": [\n"
        "    {\n"
        "      \"policy_id_guess\":\"H1\",\n"
        "      \"policy_title_guess\":\"Housing Mix\",\n"
        "      \"page_start\":12,\n"
        "      \"page_end\":15,\n"
        "      \"heading_text\":\"Policy H1: Housing Mix\",\n"
        "      \"alt_headings\":[\"H1 Housing Mix\",\"Policy H1\"],\n"
        "      \"is_ambiguous\":false,\n"
        "      \"confidence\":\"high\",\n"
        "      \"notes\":\"Why included or uncertain\"\n"
        "    }\n"
        "  ],\n"
        "  \"coverage\":[\n"
        "    {\"from_page\":1,\"to_page\":11,\"label\":\"non_policy\"}\n"
        "  ],\n"
        "  \"unassigned_headings\":[\n"
        "    {\"page\":33,\"text\":\"...\"}\n"
        "  ]\n"
        "}\n\n"
        "GUIDANCE FOR HIGH RECALL\n"
        "- Include any section that could plausibly be a policy; if unsure set is_ambiguous:true and explain in notes.\n"
        "- Prefer page ranges; char offsets not required.\n"
        "- Over-inclusion is acceptable; validation will prune later.\n\n"
        f"DOCUMENT:\n{joined}"
    )

# ========================= Index & Sweep (adaptive) ====================
def index_policies_recall(pages: List[str], model: str, base_url: str) -> Dict:
    # json_mode=False to avoid grammar slowdown during recall
    resp = chat_ollama(
        [{"role": "system", "content": SYSTEM_RECALL},
         {"role": "user", "content": build_recall_index_prompt(pages)}],
        model=model, base_url=base_url, json_mode=False
    )
    data = to_json(resp, context_label="index_policies_recall")
    data.setdefault("policies", [])
    data.setdefault("coverage", [])
    data.setdefault("unassigned_headings", [])
    return data

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

def gap_blocks(coverage: List[Dict], total_pages: int, block_size: int = GAP_BLOCK_PAGES) -> List[Dict]:
    labels = ["non_policy"] * (total_pages + 1)
    for seg in coverage:
        if seg.get("label") == "policy":
            for p in range(max(1, int(seg["from_page"])), min(total_pages, int(seg["to_page"])) + 1):
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

def _estimate_chars(pages: List[str]) -> int:
    return sum(len(p) for p in pages) + len(pages) * 16

def _merge_index_results(dst: Dict, src: Dict, offset: int) -> None:
    for p in src.get("policies", []):
        p = dict(p)
        p["page_start"] = int(p.get("page_start", 1)) + offset
        p["page_end"]   = int(p.get("page_end", 1))   + offset
        dst["policies"].append(p)
    for c in src.get("coverage", []):
        dst["coverage"].append({
            "from_page": int(c.get("from_page", 1)) + offset,
            "to_page":   int(c.get("to_page", 1))   + offset,
            "label": c.get("label", "non_policy")
        })
    for u in src.get("unassigned_headings", []):
        u = dict(u)
        u["page"] = int(u.get("page", 1)) + offset
        dst["unassigned_headings"].append(u)

def index_windowed_high_recall(pages: List[str], model: str, base_url: str) -> Dict:
    n = len(pages)
    result = {"policies": [], "coverage": [], "unassigned_headings": []}

    # seed coarse windows
    work = []
    start = 1
    while start <= n:
        end = min(n, start + INDEX_WINDOW_PAGES - 1)
        work.append((start, end))
        start = end + 1

    pbar = tqdm(total=len(work), desc="Index windows", leave=False) if len(work) > 1 else None

    while work:
        pstart, pend = work.pop(0)
        window = pages[pstart-1:pend]
        offset = pstart - 1

        try:
            idx = index_policies_recall(window, model, base_url)
            _merge_index_results(result, idx, offset)
            if pbar: pbar.update(1)
            continue
        except Exception as e:
            page_count = pend - pstart + 1
            char_count = _estimate_chars(window)
            can_split = page_count >= 3 or char_count > INDEX_MAX_CHARS // 2
            log.warning(f"[index] {pstart}-{pend} failed ({page_count}p, ~{char_count} chars): {e}")

            if can_split:
                mid = pstart + page_count // 2 - 1
                work.insert(0, (mid + 1, pend))
                work.insert(0, (pstart, mid))
                continue
            else:
                # final tiny slice: one more try; if it still fails, log and advance
                try:
                    idx = index_policies_recall(window, model, base_url)
                    _merge_index_results(result, idx, offset)
                except Exception as ee:
                    log.error(f"[index] giving up on tiny slice {pstart}-{pend}: {ee}")
                if pbar: pbar.update(1)

    if pbar: pbar.close()
    result["coverage"] = merge_coverage_segments(result["coverage"])
    return result

def sweep_gap_for_policies(pages: List[str], start_p: int, end_p: int, model: str, base_url: str) -> List[Dict]:
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
        # json_mode=False to avoid grammar slowdown here too
        resp = chat_ollama(
            [{"role": "system", "content": SYSTEM_RECALL},
             {"role": "user", "content": prompt}],
            model=model, base_url=base_url, json_mode=False
        )
        data = to_json(resp, context_label=f"gap_sweep {start_p}-{end_p}")
        return data.get("policies", []) if isinstance(data, dict) else []
    except Exception as e:
        if end_p - start_p + 1 <= 2:
            log.error(f"[gap] giving up on tiny slice {start_p}-{end_p}: {e}")
            return []
        mid = (start_p + end_p) // 2
        left = sweep_gap_for_policies(pages, start_p, mid, model, base_url)
        right = sweep_gap_for_policies(pages, mid+1, end_p, model, base_url)
        return left + right

# ========================= Extraction ==================================
def slice_pages(pages: List[str], pstart: int, pend: int) -> str:
    pstart = max(1, pstart); pend = min(len(pages), pend)
    return "\n".join([f"<<<PAGE {i}>>>\n{pages[i-1]}" for i in range(pstart, pend + 1)])

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
    # Keep grammar on for strict extraction to improve structure; turn off if too slow.
    resp = chat_ollama(
        [{"role": "system", "content": SYSTEM_STRICT},
         {"role": "user", "content": user}],
        model=model, base_url=base_url, json_mode=True
    )
    obj = to_json(resp, context_label=f"extract_policy p{pstart}-{pend}")

    if obj.get("page_start") != pstart or obj.get("page_end") != pend:
        obj.setdefault("validation_notes", []).append("Page span mismatch with requested range.")
    obj.setdefault("doc_id", doc_id)
    obj.setdefault("policy_id", pol.get("policy_id_guess", "") or "")
    obj.setdefault("policy_title", pol.get("policy_title_guess", "") or "")

    # Light quote existence check (best-effort)
    span_lower = span_text.lower()
    bad_quotes = []
    for q in obj.get("evidence_quotes", []) or []:
        qt = (q.get("quote") or "").strip()
        if qt and qt.lower() not in span_lower:
            bad_quotes.append(qt)
    if bad_quotes:
        obj["confidence"] = "low"
        obj.setdefault("validation_notes", []).append(
            f"{len(bad_quotes)} quotes not found verbatim in span"
        )
    return obj

# ========================= Streaming JSONL write =======================
def append_jsonl(out_path: str, obj: dict) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    log.info(f"[write] -> {out_path}")

# ========================= Per-PDF pipeline ============================
def process_pdf(pdf_path: str, out_dir: str, model: str, base_url: str,
                show_inner_pb: bool, preview_chars: int, dry_run: bool) -> Tuple[str, int, int]:
    """
    Returns (out_path, ok_count, total_count)
    """
    doc_id = pathlib.Path(pdf_path).stem
    pages = pdf_to_pages(pdf_path)

    if not any(pages) or all(len(p.strip()) < 10 for p in pages):
        raise RuntimeError("No extractable text (image-only or scanned). Skipping.")

    # 1) Index (adaptive)
    idx = index_windowed_high_recall(pages, model, base_url)
    policies = idx.get("policies", [])
    coverage = idx.get("coverage", [])

    # 2) Gap sweep (adaptive)
    blocks = gap_blocks(coverage, total_pages=len(pages), block_size=GAP_BLOCK_PAGES)
    for blk in blocks:
        found = sweep_gap_for_policies(pages, blk["from_page"], blk["to_page"], model, base_url)
        for pol in found:
            pol.setdefault("is_ambiguous", True)
            pol.setdefault("confidence", "low")
            pol.setdefault("notes", "Added by gap sweep (recall mode)")
        policies.extend(found)

    # Dedup a little
    dedup = {}
    for pol in policies:
        key = (int(pol.get("page_start", -1)),
               int(pol.get("page_end", -1)),
               (pol.get("heading_text") or "").strip())
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

def has_flag(argv: List[str], flag: str) -> bool:
    return flag in argv

# ========================= main =======================================
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python extract_policies.py <input.pdf|folder|glob> <out_dir> "
              "[--workers N] [--auto-pull] [--preview N] [--dry-run] "
              "[--index-window-pages N] [--gap-block-pages N]")
        sys.exit(1)

    input_path = sys.argv[1]
    out_dir = sys.argv[2]

    # Ensure output dir exists *up front*
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"[init] Output dir: {pathlib.Path(out_dir).resolve()}")

    workers = parse_flag_int(sys.argv, "--workers", DEFAULT_WORKERS)
    preview_chars = parse_flag_int(sys.argv, "--preview", 0)   # 0 = off
    dry_run = has_flag(sys.argv, "--dry-run")
    allow_autopull = has_flag(sys.argv, "--auto-pull")

    # Optional CLI overrides for window sizes
    global INDEX_WINDOW_PAGES, GAP_BLOCK_PAGES
    idx_cli = parse_flag_int(sys.argv, "--index-window-pages", INDEX_WINDOW_PAGES)
    gap_cli = parse_flag_int(sys.argv, "--gap-block-pages", GAP_BLOCK_PAGES)
    INDEX_WINDOW_PAGES = idx_cli
    GAP_BLOCK_PAGES = gap_cli

    # Model + endpoint
    wanted_model = os.getenv("OLLAMA_MODEL", MODEL_DEFAULT)  # e.g. gpt-oss:20b
    base_url = resolve_ollama_url()
    model = resolve_model_or_exit(base_url, wanted_model, allow_autopull)
    log.info(f"Using model: {model} @ {base_url}")
    log.info(f"Settings: windows={INDEX_WINDOW_PAGES} pages, max_chars={INDEX_MAX_CHARS}, gap_block={GAP_BLOCK_PAGES}, timeout={CHAT_TIMEOUT}s, generate_only={USE_GENERATE_ONLY}, llm_concurrency={LLM_MAX_CONCURRENCY}")

    pdfs = iter_pdf_paths(input_path)
    if not pdfs:
        log.error("No PDFs found.")
        sys.exit(2)

    log.info(f"Found {len(pdfs)} PDF(s). Using {workers} worker(s). Dry-run={dry_run} Preview={preview_chars}")

    results = []
    errors = 0

    with tqdm(total=len(pdfs), desc="PDFs", leave=True) as overall_pb:
        if workers <= 1 or len(pdfs) == 1:
            for pdf in pdfs:
                try:
                    out_path, ok, total = process_pdf(
                        pdf, out_dir, model, base_url,
                        show_inner_pb=True, preview_chars=preview_chars, dry_run=dry_run
                    )
                    results.append((out_path, ok, total))
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
                        False, preview_chars, dry_run
                    ): pdf for pdf in pdfs
                }
                for fut in as_completed(futs):
                    pdf = futs[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        errors += 1
                        log.exception(f"[ERROR] {pdf}: {e}")
                    finally:
                        overall_pb.update(1)

    ok_total = sum(ok for _, ok, _ in results)
    pol_total = sum(tt for _, _, tt in results)
    log.info(f"[summary] Policies extracted: {ok_total}/{pol_total}. Failures: {errors}. Outputs in: {pathlib.Path(out_dir).resolve()}")

if __name__ == "__main__":
    main()

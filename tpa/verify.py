from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .logging import log


def _read_markdown(run_dir: Path) -> str:
    md_files = list(run_dir.glob("section_*.md"))
    if not md_files:
        raise FileNotFoundError("No section markdown found in run directory")
    return md_files[0].read_text(encoding="utf-8")


def check_policy_linkage(markdown: str) -> str | None:
    paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip() and not p.strip().startswith("#")]
    for para in paragraphs:
        if "[POL:" not in para:
            return "Paragraph missing policy citation"
    return None


def check_source_diversity(markdown: str) -> str | None:
    import re

    app_ids = set(re.findall(r"\[APP:([^\]]+)\]", markdown))
    pol_ids = set(re.findall(r"\[POL:([^\]]+)\]", markdown))
    if len(app_ids | pol_ids) < 3:
        return "Fewer than 3 distinct sources cited"
    if not app_ids or not pol_ids:
        return "Missing either application or policy citations"
    return None


def check_claim_evidence(markdown: str) -> str | None:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if "[APP:" in line:
            window = " ".join(lines[idx: idx + 2])
            if "[POL:" not in window:
                return "Applicant claim without nearby policy reference"
    return None


def check_mini_balance(markdown: str) -> str | None:
    marker = "### Mini Planning Balance"
    if marker not in markdown:
        return "Missing Mini Planning Balance heading"
    section = markdown.split(marker, 1)[1].strip()
    if not section:
        return "Mini Planning Balance section empty"
    return None


def check_file_hygiene(run_dir: Path) -> str | None:
    retrieved_path = run_dir / "retrieved.json"
    if not retrieved_path.exists():
        return "retrieved.json missing"
    data = json.loads(retrieved_path.read_text(encoding="utf-8"))
    if not data:
        return "retrieved.json empty"
    for item in data:
        chunk_path = Path(item["path"])
        if not chunk_path.exists():
            return f"Missing source path: {chunk_path}"
    manifest = run_dir / "RUN.yaml"
    if manifest.exists():
        import yaml

        run_data = yaml.safe_load(manifest.read_text())
        files = run_data.get("files", {})
        reasoning_rel = files.get("reasoning")
        if not reasoning_rel:
            return "Reasoning log missing in manifest"
        reasoning_path = run_dir / reasoning_rel
        if not reasoning_path.exists():
            return "Reasoning log file missing"
        reasoning = json.loads(reasoning_path.read_text(encoding="utf-8"))
        if not reasoning.get("claims"):
            return "Reasoning log contains no claims entry"
        evidence_rel = files.get("evidence_dir")
        save_evidence = run_data.get("output", {}).get("save_zip_of_pages", True)
        if save_evidence and evidence_rel:
            evidence_dir = run_dir / evidence_rel
            if not evidence_dir.exists():
                return "Evidence directory missing"
    return None


def run_checks(run_dir: Path) -> List[str]:
    markdown = _read_markdown(run_dir)
    checks = [
        check_policy_linkage,
        check_source_diversity,
        check_claim_evidence,
        check_mini_balance,
        lambda _: check_file_hygiene(run_dir),
    ]
    failures: List[str] = []
    for check in checks:
        result = check(markdown) if check is not checks[-1] else check(None)
        if result:
            failures.append(result)
    return failures


def verify_run(run_dir: Path) -> bool:
    failures = run_checks(run_dir)
    if failures:
        for failure in failures:
            log("verify", status="fail", detail=failure, run=str(run_dir))
        return False
    log("verify", status="pass", run=str(run_dir))
    return True


__all__ = ["verify_run"]

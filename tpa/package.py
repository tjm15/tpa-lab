from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import yaml

from .logging import log


def package_run(run_dir: Path) -> Path:
    manifest_path = run_dir / "RUN.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError("RUN.yaml manifest missing; generate report first")
    manifest = yaml.safe_load(manifest_path.read_text())

    zip_path = run_dir / "package.zip"
    with ZipFile(zip_path, "w") as zf:
        for key in ["output_md", "retrieved", "prompt", "completion"]:
            rel = manifest["files"].get(key)
            if not rel:
                continue
            file_path = run_dir / rel
            if file_path.exists():
                zf.write(file_path, arcname=rel)
        evidence_dir = manifest["files"].get("evidence_dir")
        if evidence_dir:
            evidence_path = run_dir / evidence_dir
            for file in evidence_path.glob("*"):
                zf.write(file, arcname=f"{evidence_dir}/{file.name}")
    log("package", run=str(run_dir), package=str(zip_path))
    return zip_path


__all__ = ["package_run"]

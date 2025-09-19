from __future__ import annotations

"""Dataset discovery helpers for local corpora (downloads directory).

This module provides convenience functions to auto-discover the latest
Earls Court application download directory and the policies corpus so that
users do not have to manually update the YAML config each time new data is
downloaded.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re


EARLS_COURT_PREFIX = "earls-court-"


@dataclass
class CorpusPaths:
    policies: Optional[Path]
    earls_court_latest: Optional[Path]

    @property
    def existing(self) -> List[Path]:
        return [p for p in [self.policies, self.earls_court_latest] if p and p.exists()]


def _parse_timestamp_from_dirname(name: str) -> Optional[str]:
    """Extract a sortable timestamp token from a directory name.

    Expected pattern: earls-court-YYYYMMDD-HHMMSS (the trailing section may vary)
    Returns the concatenated YYYYMMDDHHMMSS string if found.
    """
    m = re.match(r"earls-court-(\d{8})-(\d{6})", name)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"  # sortable string


def latest_earls_court_dir(downloads_root: Path = Path("downloads")) -> Optional[Path]:
    """Return the latest earls-court-* directory under downloads/, if any."""
    if not downloads_root.exists():  # pragma: no cover - simple guard
        return None
    candidates = []
    for child in downloads_root.iterdir():
        if child.is_dir() and child.name.startswith(EARLS_COURT_PREFIX):
            ts = _parse_timestamp_from_dirname(child.name)
            if ts:
                candidates.append((ts, child))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def policies_dir(downloads_root: Path = Path("downloads")) -> Optional[Path]:
    cand = downloads_root / "policies"
    return cand if cand.exists() else None


def discover_corpora(downloads_root: Path = Path("downloads")) -> CorpusPaths:
    return CorpusPaths(
        policies=policies_dir(downloads_root),
        earls_court_latest=latest_earls_court_dir(downloads_root),
    )


__all__ = [
    "discover_corpora",
    "latest_earls_court_dir",
    "policies_dir",
    "CorpusPaths",
]

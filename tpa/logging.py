from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


def log(event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()


__all__ = ["log"]

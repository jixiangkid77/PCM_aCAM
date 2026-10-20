"""Run-stamp directories, logging and provenance helpers.

Run directories are never overwritten: results/<experiment>_<YYYYMMDD_HHMM>
(with the suffix _smoke for smoke runs, and _b, _c, ... if the name is
already taken).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

JST = timezone(timedelta(hours=9))


def now_jst() -> datetime:
    return datetime.now(JST)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_run_dir(results_root: Path, experiment: str, smoke: bool) -> Path:
    stamp = now_jst().strftime("%Y%m%d_%H%M")
    name = f"{experiment}_{stamp}" + ("_smoke" if smoke else "")
    run_dir = results_root / name
    suffix = ord("b")
    while run_dir.exists():
        run_dir = results_root / (name + "_" + chr(suffix))
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def environment_info() -> Dict[str, str]:
    import numpy
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def make_logger(log_path: Path, echo: bool = True) -> Callable[[str], None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{now_jst().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if echo:
            print(line, flush=True)

    return log


def write_json(path: Path, obj: Dict, indent: Optional[int] = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=False)
        f.write("\n")

"""Build a self-contained subset of the SkyWater sky130 primitive models for ngspice.

Only the two transistor flavours used by the analog-CAM bound branch are kept
(sky130_fd_pr__nfet_01v8 and sky130_fd_pr__nfet_01v8_lvt, typical corner).

Patch (the only change to the model files): the per-device mismatch hooks that the
SkyWater files leave at 0 for Spectre-style Monte Carlo
    sky130_fd_pr__<dev>__{vth0,toxe,voff,nfactor}_slope_spectre
are renamed to subcircuit parameters mm_vth0, mm_toxe, mm_voff, mm_nfactor (default 0),
so each instance can be given its own standard-normal draw, e.g.

    X1 d g s b sky130_fd_pr__nfet_01v8 w=0.42 l=0.15 mm_vth0=1.3 mm_toxe=-0.2 mm_voff=0.4

The SkyWater expressions then apply  slope / sqrt(l * w * mult) * draw  exactly as in the
PDK (Pelgrom-type scaling with the PDK's own slopes).

Source: https://github.com/google/skywater-pdk-libs-sky130_fd_pr (Apache-2.0), commit
f62031a1be9aefe902d6d54cddd6f59b57627436.

Usage: python -m pcm_acam.sky130_models <fd_pr_checkout> <out_dir>
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

DEVICES = ("nfet_01v8", "nfet_01v8_lvt")
MM_NAMES = ("vth0", "toxe", "voff", "nfactor")
SOURCE_URL = "https://github.com/google/skywater-pdk-libs-sky130_fd_pr"
SOURCE_COMMIT = "f62031a1be9aefe902d6d54cddd6f59b57627436"
LIB_NAME = "sky130_subset_tt.lib.spice"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_pm3(text: str, dev: str) -> tuple[str, dict]:
    """Rename the Spectre mismatch hooks to instance parameters; return text and counts."""
    counts = {}
    lines = text.splitlines(keepends=True)
    out = []
    for ln in lines:
        m = re.match(rf"^\.param\s+sky130_fd_pr__{dev}__(\w+)_slope_spectre\s*=\s*0(\.0)?\s*$", ln)
        if m:
            out.append("* (pcm_acam patch) global hook removed, now an instance parameter: " + ln[1:])
            continue
        out.append(ln)
    text = "".join(out)
    for name in MM_NAMES:
        token = f"sky130_fd_pr__{dev}__{name}_slope_spectre"
        body, sep, tail = text.partition("\n.subckt")
        if not sep:
            raise RuntimeError(f"{dev}: no .subckt found")
        counts[name] = tail.count(token)
        text = body + sep + tail.replace(token, f"mm_{name}")
    # add the instance parameters to the subckt default-parameter line
    pat = re.compile(r"^(\.param\s+l\s*=\s*1\s+w\s*=\s*1[^\n]*?)(\s*)$", re.M)
    n_sub = len(pat.findall(text))
    if n_sub != 1:
        raise RuntimeError(f"{dev}: expected one subckt .param line, found {n_sub}")
    text = pat.sub(lambda m: m.group(1) + " mm_vth0 = 0 mm_toxe = 0 mm_voff = 0 mm_nfactor = 0" + m.group(2), text)
    return text, counts


CORNER_NAMES = {"tt": "typical", "ss": "slow-slow", "ff": "fast-fast"}


def lib_name(corner: str) -> str:
    return f"sky130_subset_{corner}.lib.spice"


def build(fd_pr: Path, out: Path, corners: tuple = ("tt",), devices: tuple = DEVICES) -> dict:
    """Copy and patch the model files of `corners` into `out`; one .lib file per corner.

    With corners = ("tt",) the output is the subset used by E1 (third_party/sky130_fd_pr_subset).
    `devices` defaults to the two nMOS flavours of E1-E3a; E3b adds sky130_fd_pr__pfet_01v8 (same patch).
    """
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE_URL, "commit": SOURCE_COMMIT, "files": {}, "patch_counts": {}}
    if tuple(corners) != ("tt",):
        manifest["corners"] = list(corners)
    if tuple(devices) != DEVICES:
        manifest["devices"] = list(devices)
    shutil.copy2(fd_pr / "LICENSE", out / "LICENSE")
    shutil.copy2(fd_pr / "models" / "parameters" / "lod.spice", out / "lod.spice")
    for corner in corners:
        for dev in devices:
            cdir = fd_pr / "cells" / dev
            for suffix in (f"__{corner}.corner.spice", "__mismatch.corner.spice"):
                shutil.copy2(cdir / f"sky130_fd_pr__{dev}{suffix}", out / f"sky130_fd_pr__{dev}{suffix}")
            src = cdir / f"sky130_fd_pr__{dev}__{corner}.pm3.spice"
            text, counts = patch_pm3(src.read_text(), dev)
            (out / src.name).write_text(text)
            key = dev if corner == "tt" and tuple(corners) == ("tt",) and tuple(devices) == DEVICES else f"{dev}/{corner}"
            manifest["patch_counts"][key] = counts
            manifest["files"][src.name + " (original)"] = _sha(src)
        lib = out / lib_name(corner)
        lib.write_text(
            f"* Subset of the SkyWater sky130 primitive models ({CORNER_NAMES[corner]} corner) for ngspice.\n"
            f"* Source: {SOURCE_URL} commit {SOURCE_COMMIT} (Apache-2.0, see LICENSE).\n"
            f"* Devices: {', '.join('sky130_fd_pr__' + d for d in devices)}.\n"
            "* Patch: per-instance mismatch parameters mm_vth0, mm_toxe, mm_voff, mm_nfactor (see src/pcm_acam/sky130_models.py).\n"
            ".option scale=1.0u\n"
            '.include "lod.spice"\n'
            ".param lv_dlc_rotweak = 0.0 lvt_dlc_rotweak = 0.0\n"
            ".param sky130_fd_pr__nfet_01v8__dlc_rotweak = lv_dlc_rotweak\n"
            ".param sky130_fd_pr__nfet_01v8_lvt__dlc_rotweak = lvt_dlc_rotweak\n"
            + "".join(f'.include "sky130_fd_pr__{d}__mismatch.corner.spice"\n' for d in devices)
            + "".join(f'.include "sky130_fd_pr__{d}__{corner}.corner.spice"\n' for d in devices)
        )
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "MANIFEST.json":
            manifest["files"][p.name] = _sha(p)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    # python -m pcm_acam.sky130_models <fd_pr> <out> [corner ...] [--devices dev1,dev2,...]
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    args = sys.argv[3:]
    devs = DEVICES
    if "--devices" in args:
        i = args.index("--devices")
        devs = tuple(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    cs = tuple(args) or ("tt",)
    m = build(Path(sys.argv[1]), Path(sys.argv[2]), cs, devs)
    print(json.dumps(m["patch_counts"], indent=1))

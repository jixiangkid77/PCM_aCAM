"""Independent ngspice helpers for the E2 independent check (written from the protocol only).

- write_spiceinit / run_ngspice: run one netlist in batch mode (ngspice -b -r out.raw)
- parse_raw: binary raw-file reader (real data, one or more plots)
- device-line helpers for the sky130 subset models with per-instance mismatch (mm_*)

Nothing here is imported from src/ or from the main experiment scripts.
"""
import os
import subprocess
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "third_party", "sky130_fd_pr_subset_corners").replace(os.sep, "/")
DIV_MODEL = "sky130_fd_pr__nfet_01v8"
PD_MODEL = "sky130_fd_pr__nfet_01v8_lvt"
PARAMS = ("vth0", "toxe", "voff", "nfactor")

# Tight Newton tolerances for circuits with a free node (full-branch);
# harmless for single-device circuits where every terminal is driven.
TIGHT_OPTIONS = ".options reltol=1e-7 vntol=1e-10 abstol=1e-16 itl1=500 itl2=500"


def write_spiceinit(workdir):
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, ".spiceinit"), "w") as f:
        f.write("set ngbehavior=hsa\nset ng_nomodcheck\n")


def header(corner, temp_c, title="independent check", options=TIGHT_OPTIONS):
    lines = [f"* {title}",
             f'.include "{MODEL_DIR}/sky130_subset_{corner}.lib.spice"',
             f".temp {temp_c:.6g}"]
    if options:
        lines.append(options)
    return lines


def fmt(x):
    """Full-precision number for a netlist."""
    return repr(float(x))


def dev(name, d, g, s, b, model, w, l, mm=None):
    """Transistor subcircuit instance; w, l in micrometres (library sets scale=1u)."""
    txt = f"x{name} {d} {g} {s} {b} {model} w={w:g} l={l:g}"
    if mm is not None:
        for p, v in zip(PARAMS, mm):
            txt += f" mm_{p}={fmt(v)}"
    return txt


def parse_raw(path):
    """Read an ngspice binary raw file. Returns a list of plots: dict(name -> ndarray)."""
    with open(path, "rb") as f:
        data = f.read()
    plots = []
    pos = 0
    while pos < len(data):
        # header lines up to 'Binary:\n'
        nvar = npts = None
        names = []
        flags = ""
        plotname = ""
        while True:
            end = data.index(b"\n", pos)
            line = data[pos:end].decode("latin-1")
            pos = end + 1
            if line.startswith("Plotname:"):
                plotname = line.split(":", 1)[1].strip()
            elif line.startswith("Flags:"):
                flags = line.split(":", 1)[1].strip()
            elif line.startswith("No. Variables:"):
                nvar = int(line.split(":", 1)[1])
            elif line.startswith("No. Points:"):
                npts = int(line.split(":", 1)[1])
            elif line.startswith("Variables:"):
                for _ in range(nvar):
                    end = data.index(b"\n", pos)
                    parts = data[pos:end].decode("latin-1").split()
                    pos = end + 1
                    names.append(parts[1])
            elif line.startswith("Binary:"):
                break
        if "complex" in flags:
            raise ValueError("complex raw data not supported")
        nbytes = 8 * nvar * npts
        arr = np.frombuffer(data[pos:pos + nbytes], dtype="<f8").reshape(npts, nvar)
        pos += nbytes
        plot = {"__plotname__": plotname}
        for i, n in enumerate(names):
            plot[n.lower()] = arr[:, i].copy()
        plots.append(plot)
        # skip possible trailing newline between plots
        while pos < len(data) and data[pos:pos + 1] in (b"\n", b"\r"):
            pos += 1
    return plots


def run_ngspice(lines, workdir, name, timeout=3600):
    """Write <name>.sp in workdir, run ngspice in batch mode, return the parsed plots."""
    write_spiceinit(workdir)
    sp = os.path.join(workdir, name + ".sp")
    raw = os.path.join(workdir, name + ".raw")
    log = os.path.join(workdir, name + ".log")
    with open(sp, "w") as f:
        f.write("\n".join(lines) + "\n.end\n")
    if os.path.exists(raw):
        os.remove(raw)
    with open(log, "w") as lf:
        r = subprocess.run(["ngspice", "-b", "-r", raw, sp], cwd=workdir,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    txt = open(log, encoding="latin-1").read()
    bad = [w for w in ("Error", "error:", "singular", "Timestep too small",
                       "convergence failed", "doAnalyses: iteration limit")
           if w in txt]
    if r.returncode != 0 or not os.path.exists(raw) or bad:
        raise RuntimeError(f"ngspice problem in {sp}: rc={r.returncode} flags={bad}\n{txt[-3000:]}")
    return parse_raw(raw)

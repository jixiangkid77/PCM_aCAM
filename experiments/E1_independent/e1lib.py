"""Independent E1 check -- helper library (written from the E1 protocol/config only).

Everything here is computed with ngspice full simulations:
  * pull-down switching voltage V_sw: DC sweep of the gate of the LVT pull-down
    (drain 1.8 V, source/bulk 0), crossing of the drain-source current with I_sw;
  * branch bound: full branch (linear resistor G from a V_SL source to node X,
    divider nfet_01v8 from X to ground, gate = V_DL), DC sweep of V_DL, crossing
    of V(X) with V_sw.
Both use a coarse sweep (0.5 mV) followed by a fine sweep (per-branch gate
offset, 0.01 mV for branches, 0.005 mV for the pull-down) around the coarse
crossing, with linear interpolation of the crossing on the fine grid.
"""
import os
import re
import subprocess
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
LIB = os.path.join(_ROOT, "third_party", "sky130_fd_pr_subset", "sky130_subset_tt.lib.spice")
WORK = os.environ.get("E1_INDEP_WORK", os.path.join(_ROOT, "logs", "E1_independent_work"))
os.makedirs(WORK, exist_ok=True)
if not os.path.exists(os.path.join(WORK, ".spiceinit")):  # sky130 models need HSPICE compatibility
    with open(os.path.join(WORK, ".spiceinit"), "w") as _f:
        _f.write("set ngbehavior=hsa\nset ng_nomodcheck\n")

VDD = 1.8
W_DIV = 0.42
W_PD, L_PD = 0.42, 0.15
I_SW = 100e-9 * (W_PD / L_PD)          # 280 nA

# Simulator options (tight tolerances; sensitivity to these is tested separately)
OPTIONS = {"reltol": 1e-6, "abstol": 1e-15, "vntol": 1e-9}

_RUN_COUNTER = [0]
RUN_LOG = []          # (tag, n_branches, n_points, seconds)


def fmt(v):
    """Full-precision float for the netlist."""
    return repr(float(v))


def mmstr(mm):
    mm = [float(v) for v in mm]
    return ("mm_vth0=%s mm_toxe=%s mm_voff=%s mm_nfactor=%s"
            % (fmt(mm[0]), fmt(mm[1]), fmt(mm[2]), fmt(mm[3])))


def header(title, options=None):
    opts = OPTIONS if options is None else options
    lines = ["* " + title,
             '.include "%s"' % LIB,
             ".temp 27"]
    if opts:
        lines.append(".options " + " ".join("%s=%s" % (k, v) for k, v in opts.items()))
    return lines


def read_raw(path):
    """Parse an ngspice binary raw file holding a single real (DC) plot."""
    with open(path, "rb") as f:
        data = f.read()
    key = b"Binary:\n"
    idx = data.index(key)
    head = data[:idx].decode("latin-1")
    if head.count("Plotname:") != 1:
        raise RuntimeError("raw file %s holds %d plots" % (path, head.count("Plotname:")))
    flags = re.search(r"Flags:\s*(\S+)", head).group(1)
    if flags != "real":
        raise RuntimeError("unexpected raw flags %s" % flags)
    nvars = int(re.search(r"No\. Variables:\s*(\d+)", head).group(1))
    npts = int(re.search(r"No\. Points:\s*(\d+)", head).group(1))
    var_block = head.split("Variables:\n", 1)[1].strip().splitlines()
    names = [ln.split()[1].lower() for ln in var_block[:nvars]]
    arr = np.frombuffer(data[idx + len(key):], dtype="<f8", count=nvars * npts)
    arr = arr.reshape(npts, nvars)
    return {n: arr[:, k].copy() for k, n in enumerate(names)}


BAD_PATTERNS = re.compile(r"error|singular|too small|no convergence|failed|aborted|"
                          r"could not find|unknown|warning", re.IGNORECASE)


def run_spice(lines, tag):
    """Write netlist, run ngspice -b -r, return dict of vectors."""
    _RUN_COUNTER[0] += 1
    base = os.path.join(WORK, "%04d_%s" % (_RUN_COUNTER[0], tag))
    cir, raw, log = base + ".cir", base + ".raw", base + ".log"
    with open(cir, "w") as f:
        f.write("\n".join(lines) + "\n.end\n")
    if os.path.exists(raw):
        os.remove(raw)
    import time
    t0 = time.time()
    r = subprocess.run(["ngspice", "-b", "-r", raw, cir], cwd=WORK,
                       capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    out = r.stdout + "\n" + r.stderr
    with open(log, "w") as f:
        f.write(out)
    bad = [ln for ln in out.splitlines() if BAD_PATTERNS.search(ln)
           and "Total" not in ln]
    if r.returncode != 0 or bad or not os.path.exists(raw):
        raise RuntimeError("ngspice problem in %s (rc=%s):\n%s" % (cir, r.returncode, "\n".join(bad[:20])))
    res = read_raw(raw)
    npts = len(next(iter(res.values())))
    RUN_LOG.append((os.path.basename(base), len(lines), npts, dt))
    os.remove(raw)          # keep netlists and logs, drop bulky raw data
    return res


def _sweep_name(res):
    for k in ("v(v-sweep)", "v-sweep"):
        if k in res:
            return res[k]
    raise KeyError("no sweep vector in raw file: %s" % list(res)[:5])


def _branch_current(res, src):
    for k in ("i(%s)" % src, "%s#branch" % src):
        if k in res:
            return res[k]
    raise KeyError("no current vector for %s" % src)


# ----------------------------------------------------------------------------
# Pull-down switching voltage
# ----------------------------------------------------------------------------
def _first_cross_up(x, y, level):
    """First crossing of y (increasing) through level; linear interpolation."""
    above = np.nonzero(y >= level)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan, None
    k = above[0]
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0), k


def vsw_batch(mm_list, tag="pd", options=None, return_diag=False):
    """V_sw of LVT pull-downs (W/L = 0.42/0.15) with per-instance mismatch draws.

    Coarse: gate 0..1.8 V step 0.5 mV, crossing of ln(I_D) with ln(I_sw).
    Fine: per-instance gate offset, window +-0.5 mV, step 0.005 mV.
    I_D = current delivered by the 1.8 V drain source.
    """
    n = len(mm_list)
    # ---- coarse
    lines = header("pull-down V_sw coarse", options) + ["VG g 0 0"]
    for k, mm in enumerate(mm_list):
        lines.append("VD%d d%d 0 %s" % (k, k, fmt(VDD)))
        lines.append("XP%d d%d g 0 0 sky130_fd_pr__nfet_01v8_lvt w=%s l=%s %s"
                     % (k, k, W_PD, L_PD, mmstr(mm)))
    lines.append(".save " + " ".join("i(vd%d)" % k for k in range(n)))
    lines.append(".dc VG 0 1.8 0.0005")
    res = run_spice(lines, tag + "_coarse")
    vg = _sweep_name(res)
    coarse = np.empty(n)
    for k in range(n):
        idn = -_branch_current(res, "vd%d" % k)
        if np.any(np.diff(idn) <= 0):
            raise RuntimeError("pull-down %d current not strictly increasing" % k)
        lg = np.log(idn)
        coarse[k], _ = _first_cross_up(vg, lg, np.log(I_SW))
    # ---- fine
    half, step = 0.5e-3, 0.005e-3
    lines = header("pull-down V_sw fine", options) + ["VG g 0 0"]
    for k, mm in enumerate(mm_list):
        lines.append("VD%d d%d 0 %s" % (k, k, fmt(VDD)))
        lines.append("VO%d g%d g %s" % (k, k, fmt(coarse[k] - half)))
        lines.append("XP%d d%d g%d 0 0 sky130_fd_pr__nfet_01v8_lvt w=%s l=%s %s"
                     % (k, k, k, W_PD, L_PD, mmstr(mm)))
    lines.append(".save " + " ".join("i(vd%d)" % k for k in range(n)))
    lines.append(".dc VG 0 %s %s" % (fmt(2 * half), fmt(step)))
    res = run_spice(lines, tag + "_fine")
    vg = _sweep_name(res)
    fine = np.empty(n)
    for k in range(n):
        idn = -_branch_current(res, "vd%d" % k)
        lg = np.log(idn)
        v, kk = _first_cross_up(vg, lg, np.log(I_SW))
        if kk is None:
            raise RuntimeError("fine V_sw window missed for pull-down %d" % k)
        fine[k] = coarse[k] - half + v
    if return_diag:
        return fine, coarse
    return fine


# ----------------------------------------------------------------------------
# Full-branch bound
# ----------------------------------------------------------------------------
def _first_cross_down(x, y, level):
    """First crossing of y (decreasing) through level; linear interpolation."""
    below = np.nonzero(y < level)[0]
    if len(below) == 0:
        return np.inf, None            # never crosses inside the sweep
    k = below[0]
    if k == 0:
        return -np.inf, 0              # already below at the start of the sweep
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0), k


def bounds_batch(branches, tag="br", options=None, vmax=2.5, coarse_step=0.5e-3,
                 fine_half=1.0e-3, fine_step=0.01e-3, max_per_run=200, return_diag=False):
    """Bounds of full branches.

    branches: list of dicts with keys
        L    divider length (um), W fixed at 0.42 um
        mm   4 mismatch draws for the divider (vth0, toxe, voff, nfactor)
        G    PCM conductance (S)
        vsl  supply voltage V_SL (V)
        vsw  switching voltage used for the crossing (V)
    Returns array of bounds (V); +inf if V_X never falls below V_sw for V_DL <= vmax.
    """
    n = len(branches)
    out = np.empty(n)
    out_coarse = np.empty(n)
    for s in range(0, n, max_per_run):
        chunk = branches[s:s + max_per_run]
        b, bc = _bounds_chunk(chunk, "%s_%d" % (tag, s // max_per_run), options, vmax,
                              coarse_step, fine_half, fine_step)
        out[s:s + len(chunk)] = b
        out_coarse[s:s + len(chunk)] = bc
    if return_diag:
        return out, out_coarse
    return out


def _branch_lines(chunk, offsets):
    lines = []
    for k, br in enumerate(chunk):
        lines.append("VS%d s%d 0 %s" % (k, k, fmt(br["vsl"])))
        lines.append("R%d s%d x%d %s" % (k, k, k, fmt(1.0 / br["G"])))
        if offsets is None:
            gate = "g"
        else:
            gate = "g%d" % k
            lines.append("VO%d g%d g %s" % (k, k, fmt(offsets[k])))
        lines.append("XD%d x%d %s 0 0 sky130_fd_pr__nfet_01v8 w=%s l=%s %s"
                     % (k, k, gate, W_DIV, fmt(br["L"]), mmstr(br["mm"])))
    lines.append(".save " + " ".join("v(x%d)" % k for k in range(len(chunk))))
    return lines


def _bounds_chunk(chunk, tag, options, vmax, coarse_step, fine_half, fine_step):
    n = len(chunk)
    vsw = np.array([br["vsw"] for br in chunk])
    # ---- coarse: full sweep 0..vmax
    lines = header("branch bound coarse", options) + ["VDL g 0 0"]
    lines += _branch_lines(chunk, None)
    lines.append(".dc VDL 0 %s %s" % (fmt(vmax), fmt(coarse_step)))
    res = run_spice(lines, tag + "_coarse")
    vdl = _sweep_name(res)
    coarse = np.empty(n)
    for k in range(n):
        vx = res["v(x%d)" % k]
        if np.any(np.diff(vx) > 0):
            raise RuntimeError("V_X not monotonically falling for branch %d (%s)" % (k, tag))
        coarse[k], kk = _first_cross_down(vdl, vx, vsw[k])
        if kk == 0:
            raise RuntimeError("branch %d: V_X below V_sw already at V_DL = 0" % k)
    fine = coarse.copy()
    ok = np.isfinite(coarse)
    if not np.any(ok):
        return fine, coarse
    idx = np.nonzero(ok)[0]
    sub = [chunk[k] for k in idx]
    offs = coarse[idx] - fine_half
    lines = header("branch bound fine", options) + ["VDL g 0 0"]
    lines += _branch_lines(sub, offs)
    lines.append(".dc VDL 0 %s %s" % (fmt(2 * fine_half), fmt(fine_step)))
    res = run_spice(lines, tag + "_fine")
    vdl = _sweep_name(res)
    for m, k in enumerate(idx):
        vx = res["v(x%d)" % m]
        v, kk = _first_cross_down(vdl, vx, vsw[k])
        if kk is None or kk == 0 or not np.isfinite(v):
            raise RuntimeError("fine window missed the crossing for branch %d (%s)" % (k, tag))
        fine[k] = offs[m] + v
    return fine, coarse

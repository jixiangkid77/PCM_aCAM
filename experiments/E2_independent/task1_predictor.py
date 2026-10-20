"""Task 1 - independent design-time predictor (E2 protocol section 2), from the protocol only.

Per condition:
  pull-down (V_DS = 1.8 V): V_sw = V_G at I_D = 100 nA * W/L; nominal and mm_p = +-1 for each
      p in (vth0, toxe, voff, nfactor) -> dV_sw/dz_p by central differences; sigma_delta.
  divider (V_DS = V_sw,nom): nominal gate voltage V_G(I) for I = G(theta) D and a G(theta) D;
      at that V_G: s_p = -[ln I(mm_p=+1) - ln I(mm_p=-1)]/2 and
      lambda = -[ln I(V_DS+5 mV) - ln I(V_DS-5 mV)]/(10 mV)  (exact-voltage operating points)
  E[dx_supply^2] = (G/dG)^2 E[ln^2 r], r = (D - a delta)/(D - delta), delta ~ N(0, sigma_delta^2),
      trapezoid on 4001 points in [-6 sigma, 6 sigma], delta >= 0.95 D dropped, renormalised
  E[dx_dac^2]    = (G/dG)^2 {sum_p [s_p(aI) - s_p(I)]^2 + [lambda(aI) - lambda(I)]^2 sigma_delta^2}
  rms (LSB) = 15 sqrt(mean over theta_j = j/15, j = 0..15)
  settings: pdk = all terms; div_only = sum_p terms (supply 0); pd_only = lambda term for
      dac_remap and the pdk value for supply.
"""
import numpy as np
from scipy import integrate

import ind_spice as S
import ind_core as C

PARAMS = S.PARAMS
DG = 20e-6


def g_of_x(x):
    return (2.5 + 20.0 * np.asarray(x, float)) * 1e-6


def drift_factor(t_s, nu=0.049, t0=20.0):
    return (t_s / t0) ** (-nu)


def pulldown_sensitivity(cond, pd, workdir):
    i_target = 100e-9 * pd["w_um"] / pd["l_um"]
    devs, labels = [], []
    base = dict(model=S.PD_MODEL, w=pd["w_um"], l=pd["l_um"], vds=1.8)
    devs.append(dict(base, mm=[0, 0, 0, 0])); labels.append("nom")
    for i, p in enumerate(PARAMS):
        for sgn in (+1, -1):
            mm = [0, 0, 0, 0]; mm[i] = sgn
            devs.append(dict(base, mm=mm)); labels.append(f"{p}{'+' if sgn > 0 else '-'}")
    r = C.solve_vg(cond["corner"], cond["temp_c"], devs, [i_target] * len(devs),
                   workdir, f"{cond['id']}_pd_sens")
    v = dict(zip(labels, r["root"]))
    out = {"vsw_nom": v["nom"], "i_target_A": i_target}
    for p in PARAMS:
        out[f"dvsw_d{p}"] = (v[p + "+"] - v[p + "-"]) / 2.0
    out["sigma_delta"] = float(np.sqrt(sum(out[f"dvsw_d{p}"] ** 2 for p in PARAMS)))
    out["vsw_lin_nom"] = float(r["coarse_lin"][0])
    out["max_abs_resid_lnI"] = float(np.max(np.abs(r["resid_lnI"])))
    out["refine_shift_max_V"] = float(np.max(np.abs(r["stage2_minus_stage1"])))
    return out


def divider_sensitivity(cond, vsw_nom, a, workdir):
    """s_p and lambda at the nominal gate voltage for I = G D and a G D, all 16 thetas."""
    D = cond["d_v"]
    theta = np.arange(16) / 15.0
    G = g_of_x(theta)
    I_lo = G * D           # current at programming
    I_aI = a * G * D       # drifted current
    targets = np.concatenate([I_lo, I_aI])
    base = dict(model=S.DIV_MODEL, w=cond["w_div_um"], l=cond["l_div_um"], vds=vsw_nom,
                mm=[0, 0, 0, 0])
    tag = f"{cond['id']}_div"
    r = C.solve_vg(cond["corner"], cond["temp_c"], [base] * len(targets), targets,
                   workdir, tag + "_vg")
    vg = r["root"]
    # operating points: nominal, mm_p = +-1, V_DS +- 5 mV, at each gate voltage
    variants = [("nom", [0, 0, 0, 0], vsw_nom)]
    for i, p in enumerate(PARAMS):
        for sgn in (+1, -1):
            mm = [0, 0, 0, 0]; mm[i] = sgn
            variants.append((f"{p}{'+' if sgn > 0 else '-'}", mm, vsw_nom))
    variants.append(("vds+", [0, 0, 0, 0], vsw_nom + 0.005))
    variants.append(("vds-", [0, 0, 0, 0], vsw_nom - 0.005))
    devs = []
    for (lab, mm, vds) in variants:
        for k in range(len(vg)):
            devs.append(dict(model=S.DIV_MODEL, w=cond["w_div_um"], l=cond["l_div_um"],
                             mm=mm, vds=vds, vg=vg[k]))
    Iop = C.op_currents(cond["corner"], cond["temp_c"], devs, workdir, tag + "_ops")
    Iop = Iop.reshape(len(variants), len(vg))
    lnI = {lab: np.log(Iop[i]) for i, (lab, _, _) in enumerate(variants)}
    sens = {}
    for p in PARAMS:
        sens[p] = -(lnI[p + "+"] - lnI[p + "-"]) / 2.0
    sens["lambda"] = -(lnI["vds+"] - lnI["vds-"]) / 0.010
    n = len(theta)
    s_I = {k: v[:n] for k, v in sens.items()}
    s_aI = {k: v[n:] for k, v in sens.items()}
    diag = dict(vg_I=vg[:n], vg_aI=vg[n:],
                nominal_resid_lnI_max=float(np.max(np.abs(lnI["nom"] - np.log(targets)))),
                refine_shift_max_V=float(np.max(np.abs(r["stage2_minus_stage1"]))))
    return theta, G, s_I, s_aI, diag


def e_ln2_r(D, a, sigma, n=4001, nsig=6.0):
    d = np.linspace(-nsig * sigma, nsig * sigma, n)
    keep = d < 0.95 * D
    d = d[keep]
    w = np.exp(-0.5 * (d / sigma) ** 2)
    f = np.log((D - a * d) / (D - d)) ** 2
    trap = getattr(np, "trapezoid", None) or np.trapz
    val = trap(w * f, d) / trap(w, d)
    return float(val), int((~keep).sum())


def e_ln2_r_quad(D, a, sigma):
    """Reference value by adaptive quadrature (diagnostic only)."""
    pdf = lambda d: np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    f = lambda d: np.log((D - a * d) / (D - d)) ** 2 * pdf(d)
    lo, hi = -6 * sigma, min(6 * sigma, 0.95 * D)
    num = integrate.quad(f, lo, hi, epsabs=0, epsrel=1e-12, limit=200)[0]
    den = integrate.quad(pdf, lo, hi, epsabs=0, epsrel=1e-12, limit=200)[0]
    return num / den


def predict_condition(cond, pulldowns, t_s, workdir):
    pd = pulldowns[cond["pulldown"]]
    a = drift_factor(t_s)
    pds = pulldown_sensitivity(cond, pd, workdir)
    theta, G, s_I, s_aI, diag = divider_sensitivity(cond, pds["vsw_nom"], a, workdir)
    sig = pds["sigma_delta"]
    D = cond["d_v"]
    k2 = (G / DG) ** 2
    div_term = sum((s_aI[p] - s_I[p]) ** 2 for p in PARAMS)
    pd_term = (s_aI["lambda"] - s_I["lambda"]) ** 2 * sig ** 2
    eln2, ndrop = e_ln2_r(D, a, sig)
    ms = {
        "pdk|dac_remap": k2 * (div_term + pd_term),
        "pdk|supply": k2 * eln2,
        "div_only|dac_remap": k2 * div_term,
        "div_only|supply": np.zeros_like(k2),
        "pd_only|dac_remap": k2 * pd_term,
        "pd_only|supply": k2 * eln2,
    }
    rms = {k: float(15.0 * np.sqrt(np.mean(v))) for k, v in ms.items()}
    return dict(a=a, pulldown=pds, theta=theta, G=G, s_at_I=s_I, s_at_aI=s_aI, diag=diag,
                e_ln2_r=eln2, e_ln2_r_quad=e_ln2_r_quad(D, a, sig), n_dropped=ndrop,
                per_theta_ms_x=ms, rms_lsb=rms,
                vsw_nom_V=pds["vsw_nom"], v_sl_nom_V=pds["vsw_nom"] + D)

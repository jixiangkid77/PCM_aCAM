"""Task 2 - full-branch Monte Carlo spot check (E2 protocol section 4), from the protocol only.

Per condition, setting pdk (both devices with PDK mismatch), instances k = 0..n-1, drift 1 year:
  draws        rng = default_rng([20261014, 2]); z_div = rng.standard_normal((300, 4)), then
               z_pd = rng.standard_normal((300, 4)); columns (vth0, toxe, voff, nfactor)
  V_sw,k       pull-down (z_pd[k]) gate voltage at 100 nA * W/L, V_DS = 1.8 V
  V_SL,nom     V_sw,nom + D
  M0(theta)    nominal full branch: G(theta), SL = V_SL,nom, bound where V_X = V_sw,nom
  programming  G_k(theta) = I_div,k(V_G = M0(theta), V_DS = V_sw,k) / (V_SL,nom - V_sw,k)
  drift        G -> a G_k, a = (31536000/20)^-0.049
  dac_remap    bound B: instance full branch (a G_k, SL = V_SL,nom), V_X = V_sw,k;
               x_eff solves Ma(x) = B_nom(a G(x)) = B, i.e. a G(x_eff) D = I_nom(B, V_DS = V_sw,nom)
  supply       bound B: instance full branch (a G_k, SL = V_sw,nom + D/a), V_X = V_sw,k;
               x_eff solves M0(x) = B_nom(G(x)) = B, i.e. G(x_eff) D = I_nom(B, V_DS = V_sw,nom)
  error        15 (x_eff - theta)  [LSB]
  closure      nominal full branch with the conductance a G(x_eff) (dac_remap) or G(x_eff)
               (supply), SL = V_SL,nom, threshold V_sw,nom, must reproduce B (diagnostic)
A zero-mismatch instance is appended as a pipeline self-check (its errors must be 0).
"""
import numpy as np

import ind_spice as S
import ind_core as C
from task1_predictor import g_of_x, drift_factor

SEED = [20261014, 2]
N_TOTAL = 300


def draws():
    rng = np.random.default_rng(SEED)
    z_div = rng.standard_normal((N_TOTAL, 4))
    z_pd = rng.standard_normal((N_TOTAL, 4))
    return z_div, z_pd


def run_condition(cond, pulldowns, t_s, workdir, n_inst=40, theta_idx=(1, 5, 10, 14)):
    cid, corner, temp = cond["id"], cond["corner"], cond["temp_c"]
    pd = pulldowns[cond["pulldown"]]
    D = cond["d_v"]
    a = drift_factor(t_s)
    z_div, z_pd = draws()
    # instances 0..n_inst-1 plus one zero-mismatch self-check instance (last row)
    zd = np.vstack([z_div[:n_inst], np.zeros((1, 4))])
    zp = np.vstack([z_pd[:n_inst], np.zeros((1, 4))])
    n = zd.shape[0]
    i_target = 100e-9 * pd["w_um"] / pd["l_um"]

    # pull-down switching voltages: nominal + instances
    pdevs = [dict(model=S.PD_MODEL, w=pd["w_um"], l=pd["l_um"], vds=1.8, mm=[0, 0, 0, 0])]
    pdevs += [dict(model=S.PD_MODEL, w=pd["w_um"], l=pd["l_um"], vds=1.8, mm=list(zp[k]))
              for k in range(n)]
    r = C.solve_vg(corner, temp, pdevs, [i_target] * len(pdevs), workdir, f"{cid}_mc_vsw")
    vsw_nom = r["root"][0]
    vsw = r["root"][1:]
    vsl_nom = vsw_nom + D
    vsl_sup = vsw_nom + D / a

    # nominal map M0(theta_j), all 16 thresholds, full branch
    theta_all = np.arange(16) / 15.0
    G_all = g_of_x(theta_all)
    m0br = [dict(G=g, vsl=vsl_nom, thr=vsw_nom, model=S.DIV_MODEL, w=cond["w_div_um"],
                 l=cond["l_div_um"], mm=[0, 0, 0, 0]) for g in G_all]
    m0 = C.branch_bounds(corner, temp, m0br, workdir, f"{cid}_mc_m0")
    M0 = m0["bound"]

    # threshold-verify programming (no clipping)
    tj = np.array(theta_idx)
    th = theta_all[tj]
    devs = []
    for k in range(n):
        for j in tj:
            devs.append(dict(model=S.DIV_MODEL, w=cond["w_div_um"], l=cond["l_div_um"],
                             mm=list(zd[k]), vg=M0[j], vds=vsw[k]))
    Iprog = C.op_currents(corner, temp, devs, workdir, f"{cid}_mc_prog").reshape(n, len(tj))
    Gk = Iprog / (vsl_nom - vsw)[:, None]

    # drifted bounds, full branch
    nodes = ("dac_remap", "supply")
    brs, keys = [], []
    for node, vsl in (("dac_remap", vsl_nom), ("supply", vsl_sup)):
        for k in range(n):
            for jj in range(len(tj)):
                brs.append(dict(G=a * Gk[k, jj], vsl=vsl, thr=vsw[k], model=S.DIV_MODEL,
                                w=cond["w_div_um"], l=cond["l_div_um"], mm=list(zd[k])))
                keys.append((node, k, jj))
    bb = C.branch_bounds(corner, temp, brs, workdir, f"{cid}_mc_drift")
    B = np.empty((2, n, len(tj)))
    for (node, k, jj), b in zip(keys, bb["bound"]):
        B[nodes.index(node), k, jj] = b

    # input-referred error via the nominal maps (exact-voltage nominal current at B)
    ndevs = [dict(model=S.DIV_MODEL, w=cond["w_div_um"], l=cond["l_div_um"], mm=[0, 0, 0, 0],
                  vg=b, vds=vsw_nom) for b in B.ravel()]
    Inom = C.op_currents(corner, temp, ndevs, workdir, f"{cid}_mc_map").reshape(B.shape)
    Gp = Inom / D                        # conductance whose nominal bound is B
    x_eff = np.empty_like(B)
    x_eff[0] = (Gp[0] / a - 2.5e-6) / 20e-6      # dac_remap: a G(x) = Gp
    x_eff[1] = (Gp[1] - 2.5e-6) / 20e-6          # supply:    G(x) = Gp
    err = 15.0 * (x_eff - th[None, None, :])

    # closure: nominal full branch at the conductance Gp must reproduce the bound B
    cbr = [dict(G=g, vsl=vsl_nom, thr=vsw_nom, model=S.DIV_MODEL, w=cond["w_div_um"],
                l=cond["l_div_um"], mm=[0, 0, 0, 0]) for g in Gp.ravel()]
    cl = C.branch_bounds(corner, temp, cbr, workdir, f"{cid}_mc_closure")
    closure = cl["bound"].reshape(B.shape) - B
    # error-equivalent of the closure residual: d err / d B ~ 15 * gm/(D * 20 uS) [/a]
    return dict(a=a, vsw_nom=vsw_nom, vsl_nom=vsl_nom, vsl_supply=vsl_sup,
                vsw=vsw[:n_inst], vsw_selfcheck=vsw[n_inst], theta_all=theta_all, M0=M0,
                theta_idx=tj, theta=th, Gk=Gk[:n_inst], bound=B[:, :n_inst], x_eff=x_eff[:, :n_inst],
                err=err[:, :n_inst], err_selfcheck=err[:, n_inst],
                closure_max_abs_V=float(np.max(np.abs(closure))),
                diag=dict(vsw_resid_lnI_max=float(np.max(np.abs(r["resid_lnI"]))),
                          m0_op_resid_lnI_max=float(np.max(np.abs(m0["op_resid_lnI"]))),
                          drift_op_resid_lnI_max=float(np.max(np.abs(bb["op_resid_lnI"]))),
                          drift_refine_shift_max_V=float(np.max(np.abs(bb["stage2_minus_stage1"]))),
                          drift_dvx_dvdl_range=[float(bb["dvx_dvdl"].min()), float(bb["dvx_dvdl"].max())],
                          closure_op_resid_lnI_max=float(np.max(np.abs(cl["op_resid_lnI"])))),
                nodes=nodes)

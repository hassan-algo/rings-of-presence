"""
Every number and figure in the paper "Rings of Presence: one balance rule for gravitational
boundaries and a geometric factor linking the galactic acceleration scale to Lambda".

Usage:
  python3 paper_analysis.py <SPARC Rotmod_LTG folder> <local_volume_database folder> <figure output folder>

Parts
  A. The two predicted values of a0 = c^2 sqrt(Lambda/3) / k,  k = pi (1 + x)  (planar x = 1, spherical x = 1/2)
  B. SPARC: global a0 (two estimators), bootstrap over galaxies, stellar mass-to-light systematics,
     gas-rich subsample, galaxy-by-galaxy fits and the size test, likelihood comparison with intrinsic scatter
  C. Ownership radius r^3 = G m / S: outermost moons, satellite galaxies, the Local Group zero-velocity
     radius from field-galaxy velocities, and the Lambda limit for the Local Group and Virgo
  D. Figures (PDF)
"""
import sys, os, csv, math
import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_a0 import load, g_pred

SPARC, LVDB, FIGDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(FIGDIR, exist_ok=True)

KMS2_PER_KPC = 1e6 / 3.0856776e19
C = 299_792_458.0
MPC = 3.0856776e22
G_SI = 6.674e-11
MSUN = 1.989e30
AU_KM = 1.495979e8
COSMO = {"Planck": (67.4, 0.685), "SH0ES": (73.0, 0.70)}


def a_lambda(H0, OL):
    """c^2 sqrt(Lambda/3) = c H0 sqrt(Omega_Lambda)  [m/s^2]"""
    return C * H0 * 1e3 / MPC * np.sqrt(OL)


def lam(H0, OL):
    """Lambda in m^-2 from Lambda c^2 / 3 = H0^2 Omega_Lambda"""
    return 3 * (H0 * 1e3 / MPC) ** 2 * OL / C**2


print("A. Predicted a0 = a_Lambda / k")
PRED = {}
for lab, (H0, OL) in COSMO.items():
    aL = a_lambda(H0, OL)
    PRED[lab] = {"aL": aL, "planar": aL / (2 * np.pi), "spherical": aL / (1.5 * np.pi)}
    print(f"  {lab:6s} H0={H0} OmegaL={OL}: a_Lambda = {aL:.4e};  planar (k=2pi) {aL/(2*np.pi):.4e};  "
          f"spherical (k=3pi/2) {aL/(1.5*np.pi):.4e}   Lambda = {lam(H0, OL):.3e} m^-2")
# k from the presence integral: total presence of a point source inside a ring of radius R,
# divided by its presence at the edge, in units of R^n.  k = S_(n-1) / (n - 1)
from math import gamma
for n in (2, 3, 4):
    Sn = 2 * np.pi**(n / 2) / gamma(n / 2)
    print(f"  n = {n}: k = {Sn/(n-1):.4f} = {Sn/(n-1)/np.pi:.4f} pi   (earlier form pi*n/(n-1) = {np.pi*n/(n-1):.4f})")

# ------------------------------------------------------------------ B. SPARC
names, arr = load(SPARC)
R, Vobs, eV, Vgas, Vdisk, Vbul = arr.T
sq = lambda v: v * np.abs(v)
gobs = Vobs**2 / R * KMS2_PER_KPC
fe = eV / Vobs
sigv = np.maximum(2 * fe / np.log(10), 0.02)          # dex, from velocity errors only
def gbar(ud, ub=None):
    ub = 1.4 * ud if ub is None else ub
    return (sq(Vgas) + ud * sq(Vdisk) + ub * sq(Vbul)) / R * KMS2_PER_KPC
good = fe < 0.1
print(f"\nB. SPARC: {len(R)} points in {len(set(names))} galaxies; after velocity-error cut "
      f"{np.sum(good)} points in {len(set(names[good]))} galaxies")

GRID = np.geomspace(0.3e-10, 4e-10, 2400)
def resid(sel, ud, a):
    gb = gbar(ud); k = sel & good & (gb > 0)
    return np.log10(gobs[k]) - np.log10(g_pred(gb[k], a)), k
def best_rms(sel, ud, w=None):
    gb = gbar(ud); k = sel & good & (gb > 0)
    lo, ww = np.log10(gobs[k]), (np.ones(k.sum()) if w is None else w[k])
    m = [np.sum(ww * (lo - np.log10(g_pred(gb[k], a)))**2) / ww.sum() for a in GRID]
    i = int(np.argmin(m)); return GRID[i], math.sqrt(m[i])
def best_chi(sel, ud, sint=0.0, w=None):
    gb = gbar(ud); k = sel & good & (gb > 0)
    lo, s2 = np.log10(gobs[k]), sigv[k]**2 + sint**2
    ww = np.ones(k.sum()) if w is None else w[k]
    c = [np.sum(ww * (lo - np.log10(g_pred(gb[k], a)))**2 / s2) for a in GRID]
    i = int(np.argmin(c)); return GRID[i], c[i], k.sum()

ALL = np.ones(len(R), bool)
gal_all = sorted(set(names[good]))
a_rms, rms = best_rms(ALL, 0.5)
a_chi, chi_min, npt = best_chi(ALL, 0.5)
print(f"  global, Upsilon_disk 0.5 / bulge 0.7: rms estimator a0 = {a_rms:.4e} (scatter {rms:.3f} dex); "
      f"chi2 estimator (velocity errors only) a0 = {a_chi:.4e}, chi2 = {chi_min:.0f} for {npt} points")

rng = np.random.default_rng(7)
def bootstrap(sel_gal, ud, est, n=600):
    out = []
    for _ in range(n):
        pick = rng.choice(sel_gal, len(sel_gal))
        w = np.zeros(len(R))
        for p in pick: w += (names == p)
        sel = w > 0
        out.append(est(sel, ud, w)[0])
    return np.percentile(out, [16, 50, 84])
b_lo, b_me, b_hi = bootstrap(gal_all, 0.5, lambda s, u, w: best_rms(s, u, w))
print(f"  bootstrap over galaxies (rms estimator): a0 = {b_me:.3e} (+{b_hi-b_me:.2e}/-{b_me-b_lo:.2e})")
UPS = (0.3, 0.4, 0.5, 0.6, 0.7)
ups_all = {u: best_rms(ALL, u)[0] for u in UPS}
print("  stellar mass-to-light systematics (rms estimator):",
      "; ".join(f"Ud={u}: {a:.3e}" for u, a in ups_all.items()))

# intrinsic scatter that makes chi2/dof = 1 at the best fit
def sint_for(sel, ud):
    f = lambda s: best_chi(sel, ud, s)[1] / (best_chi(sel, ud, s)[2] - 1) - 1
    return brentq(f, 0.0, 0.5, xtol=1e-4)
s_int = sint_for(ALL, 0.5)
a_chi_s, chi_s, n_s = best_chi(ALL, 0.5, s_int)
print(f"  intrinsic scatter for chi2/dof = 1: {s_int:.3f} dex -> best a0 = {a_chi_s:.4e}")

# gas-rich subsample
gb05 = gbar(0.5); gas_rich = np.zeros(len(R), bool); gas_names = []
for g in sorted(set(names)):
    k = (names == g) & (gb05 > 0) & (gb05 < 3e-10)
    if k.sum() < 4: continue
    share = np.mean(np.maximum(sq(Vgas[k]), 0) / np.maximum(sq(Vgas[k]) + 0.5 * sq(Vdisk[k]) + 0.7 * sq(Vbul[k]), 1e-9))
    if share > 0.6: gas_rich |= (names == g); gas_names.append(g)
print(f"  gas-rich galaxies (gas > 60% of baryonic acceleration where g_bar < 3e-10): {len(gas_names)}: {', '.join(gas_names)}")
ups_gas = {u: best_chi(gas_rich, u)[0] for u in (0.25, 0.5, 0.75)}
print("  gas-rich chi2 estimator:", "; ".join(f"Ud={u}: {a:.3e}" for u, a in ups_gas.items()))
ups_gas_rms = {u: best_rms(gas_rich, u)[0] for u in (0.25, 0.5, 0.75)}
print("  gas-rich rms estimator:", "; ".join(f"Ud={u}: {a:.3e}" for u, a in ups_gas_rms.items()))
g_lo, g_me, g_hi = bootstrap(gas_names, 0.5, lambda s, u, w: best_chi(s, u, 0.0, w))
print(f"  gas-rich bootstrap over galaxies (chi2 estimator): a0 = {g_me:.3e} (+{g_hi-g_me:.2e}/-{g_me-g_lo:.2e})")
s_int_gas = sint_for(gas_rich, 0.5)
print(f"  gas-rich intrinsic scatter {s_int_gas:.3f} dex -> best a0 {best_chi(gas_rich, 0.5, s_int_gas)[0]:.3e}")

# likelihood comparison of the four candidate values, with intrinsic scatter included
print("  chi2 at candidate a0 (velocity errors + intrinsic scatter fitted to each sample):")
CANDS = [("planar, Planck", PRED["Planck"]["planar"]), ("planar, SH0ES", PRED["SH0ES"]["planar"]),
         ("spherical, Planck", PRED["Planck"]["spherical"]), ("spherical, SH0ES", PRED["SH0ES"]["spherical"])]
for lab, sel, si in (("all", ALL, s_int), ("gas-rich", gas_rich, s_int_gas)):
    best = best_chi(sel, 0.5, si)
    vals = []
    for cl, a in CANDS:
        r, k = resid(sel, 0.5, a)
        c = np.sum(r**2 / (sigv[k]**2 + si**2)); vals.append((cl, c - best[1]))
    print(f"    {lab:9s} best a0 {best[0]:.3e}:  " + ";  ".join(f"{cl}: dchi2 = {d:+.1f}" for cl, d in vals))

# galaxy-by-galaxy fits and the size test
rows = []
gb = gbar(0.5)
for g in sorted(set(names)):
    kk = names == g; k = kk & (gb > 0) & good
    if k.sum() < 8: continue
    lo = np.log10(gobs[k]); s = sigv[k]
    ch = np.array([np.sum(((lo - np.log10(g_pred(gb[k], a))) / s)**2) for a in GRID])
    i = int(ch.argmin())
    if i in (0, len(GRID) - 1): continue
    ok = GRID[ch <= ch[i] + 1]
    Rpk = R[kk][np.argmax(np.abs(Vdisk[kk]))]
    rows.append((g, GRID[i], np.log10(ok.max() / ok.min()) / 2, Rpk / 2.2, R[kk].max()))
a0g = np.array([r[1] for r in rows]); Rd = np.array([r[3] for r in rows]); Rmax = np.array([r[4] for r in rows])
errg = np.array([r[2] for r in rows])
rho_d, p_d = spearmanr(Rd, a0g); rho_m, p_m = spearmanr(Rmax, a0g)
xneed = PRED["Planck"]["aL"] / (np.pi * a0g) - 1
print(f"  galaxy-by-galaxy: {len(rows)} galaxies; median a0 {np.median(a0g):.3e}; scatter {np.std(np.log10(a0g)):.2f} dex; "
      f"median fit error {np.median(errg):.3f} dex")
print(f"    a0 vs disk scale length: Spearman {rho_d:+.2f} (p = {p_d:.2f}); vs outermost radius: {rho_m:+.2f} (p = {p_m:.2f})")
print(f"    x required (Planck): median {np.median(xneed):.2f}, 16-84%: {np.percentile(xneed,16):.2f} to {np.percentile(xneed,84):.2f}")

# ------------------------------------------------------------------ C. ownership radius
print("\nC. Ownership radius r^3 = G m / S")
planets = [("Earth", 1.000, 5.972e24, "Moon", 384_400, False), ("Mars", 1.524, 6.417e23, "Deimos", 23_457, False),
           ("Jupiter", 5.204, 1.898e27, "Kore", 24_203_300, True), ("Saturn", 9.583, 5.683e26, "S/2004 S 52", 26_446_400, True),
           ("Uranus", 19.19, 8.681e25, "Ferdinand", 20_421_400, True), ("Neptune", 30.07, 1.024e26, "S/2021 N 1", 50_700_200, True)]
moons = []
for n, a, m, moon, am, retro in planets:
    RH = a * AU_KM * (m / (3 * MSUN)) ** (1 / 3)
    moons.append((n, moon, RH, am / RH, retro))
    print(f"  {n:8s} r = {RH/1e6:7.2f} Mkm; outermost {moon:12s} at {am/RH:.3f} r ({'retrograde' if retro else 'prograde'})")

rd = lambda f: list(csv.DictReader(open(os.path.join(LVDB, "data", f))))
num = lambda s: float(s) if s not in ("", "nan", None) else float("nan")
def eggleton(q):
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + math.log(1 + q ** (1 / 3)))
D_MW_M31 = 780.0
sats = {}
for host, col, mh, mo, f in (("Milky Way", "distance_gc", 1.0e12, 1.5e12, "dwarf_mw.csv"),
                             ("Andromeda", "distance_m31", 1.5e12, 1.0e12, "dwarf_m31.csv")):
    ring = eggleton(mh / mo) * D_MW_M31
    d = np.array([num(r[col]) for r in rd(f)]); d = d[np.isfinite(d)]
    sats[host] = (ring, d)
    out = np.sort(d[d > ring])
    print(f"  {host}: ring {ring:.0f} kpc; {len(d)} satellites; beyond: {len(out)} at "
          f"{', '.join(f'{x:.0f} ({(x/ring-1)*100:.0f}%)' for x in out)}")
print(f"  total inside: {sum(np.sum(d <= r) for r, d in sats.values())} of {sum(len(d) for r, d in sats.values())}")

rows_all = rd("dwarf_all.csv")
rl = np.array([num(x["distance_lg"]) for x in rows_all]) / 1000.0
vl = np.array([num(x["velocity_lg"]) for x in rows_all])
dgc = np.array([num(x["distance_gc"]) for x in rows_all])
dm31 = np.array([num(x["distance_m31"]) for x in rows_all])
nm = np.array([x["name"] for x in rows_all])
base_sel = np.isfinite(rl) & np.isfinite(vl) & (rl > 0.35) & (rl < 3.0)
field = base_sel & ~((dgc < 300) | (dm31 < 330))
print(f"  Local Group: {base_sel.sum()} galaxies at 0.35-3 Mpc; {field.sum()} after removing satellites "
      f"(within 300 kpc of the Milky Way or 330 kpc of M31)")
model = lambda rr, H, R0: H * rr * (1 - (R0 / rr) ** 2)
r, v = rl[field], vl[field]
p, pc = curve_fit(model, r, v, p0=(70, 1.0))
bs = []
for _ in range(2000):
    i = rng.integers(0, len(r), len(r))
    try: bs.append(curve_fit(model, r[i], v[i], p0=p)[0][1])
    except Exception: pass
lo, me, hi = np.percentile(bs, [16, 50, 84])
print(f"  fit v = H r (1 - (R0/r)^2): R0 = {p[1]:.3f} Mpc (bootstrap {me:.2f} +{hi-me:.2f}/-{me-lo:.2f}), H = {p[0]:.0f} km/s/Mpc, "
      f"rms {np.std(v - model(r, *p)):.0f} km/s")
o = np.argsort(r)
last_neg = [(nm[field][j], r[j], v[j]) for j in o if v[j] < 0][-1]
first_pos_after = [(nm[field][j], r[j], v[j]) for j in o if v[j] > 0 and r[j] > last_neg[1]][0]
print(f"  outermost infalling field galaxy: {last_neg[0]} at {last_neg[1]:.2f} Mpc ({last_neg[2]:+.0f} km/s); "
      f"next receding: {first_pos_after[0]} at {first_pos_after[1]:.2f} Mpc ({first_pos_after[2]:+.0f} km/s)")
LamP = lam(*COSMO["Planck"])
RL = lambda M: (3 * G_SI * M * MSUN / (LamP * C**2)) ** (1 / 3) / MPC
for M in (1.9e12, 3e12, 5e12):
    print(f"  Lambda limit R_L for M = {M:.1e} Msun: {RL(M):.2f} Mpc")
for M in (5.7e14, 8.0e14, 10.3e14):
    print(f"  Virgo: Lambda limit for M = {M:.2e} Msun: {RL(M):.2f} Mpc   (measured zero-velocity radius 7.2 +/- 0.7 Mpc)")

# ------------------------------------------------------------------ D. figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
BLUE, ORANGE, GREY, RED, GREEN = "#2c5d8a", "#d9822b", "#8a8a8a", "#b23b3b", "#3c8d5a"

# Figure 1: ownership radius
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.25], "wspace": 0.38})
a = ax[0]
xs = np.arange(len(moons))
a.bar(xs, [m[3] for m in moons], color=[GREY if not m[4] else BLUE for m in moons], width=0.6)
a.axhline(1, color="k", lw=0.8); a.axhline(0.5, color=GREY, ls="--", lw=0.7); a.axhline(0.7, color=BLUE, ls=":", lw=0.8)
a.text(5.45, 1.02, "ring edge", ha="right", va="bottom", fontsize=7)
a.text(-0.4, 0.52, "prograde limit", fontsize=6.5, color=GREY, va="bottom")
a.text(-0.4, 0.72, "retrograde limit", fontsize=6.5, color=BLUE, va="bottom")
a.set_xticks(xs, [m[0] for m in moons], rotation=45, ha="right")
a.set_ylim(0, 1.12); a.set_ylabel(r"outermost moon $a\,/\,r_{\rm own}$"); a.set_title("(a) planets", fontsize=8.5)
a = ax[1]
bins = np.linspace(0, 1.4, 29)
for (host, (ring, d)), col in zip(sats.items(), (BLUE, ORANGE)):
    a.hist(d / ring, bins=bins, histtype="step", lw=1.1, color=col, label=f"{host} ({len(d)})")
a.axvline(1, color="k", lw=0.8)
a.set_xlabel(r"satellite distance $/\,r_{\rm own}$"); a.set_ylabel("number")
a.legend(frameon=False, loc="upper left"); a.set_title("(b) satellite galaxies", fontsize=8.5)
a = ax[2]
a.scatter(rl[base_sel & ~field], vl[base_sel & ~field], s=5, color=GREY, alpha=0.5, lw=0, label="satellites (excluded)")
a.scatter(r, v, s=9, color=BLUE, lw=0, label=f"field galaxies ({field.sum()})")
rr = np.linspace(0.35, 3, 200); a.plot(rr, model(rr, *p), color="k", lw=0.9)
a.axhline(0, color=GREY, lw=0.5)
a.axvspan(lo, hi, color=BLUE, alpha=0.15, lw=0); a.axvline(0.96, color=GREEN, ls="--", lw=0.8)
a.axvspan(RL(1.9e12), RL(5e12), color=ORANGE, alpha=0.18, lw=0)
a.text((RL(1.9e12) + RL(5e12)) / 2, 385, r"$R_\Lambda$ limit", color=ORANGE, ha="center", va="top", fontsize=7.5)
a.text(0.99, 395, "K09", color=GREEN, fontsize=6.5, va="top")
a.set_xlim(0.3, 3.0); a.set_ylim(-280, 420)
a.set_xlabel(r"distance from Local Group centre [Mpc]"); a.set_ylabel(r"$v_{\rm LG}$ [km s$^{-1}$]")
a.legend(frameon=False, loc="lower right", markerscale=1.5, handletextpad=0.2); a.set_title("(c) Local Group", fontsize=8.5)
fig.savefig(os.path.join(FIGDIR, "fig_rings.pdf")); plt.close(fig)

# Upsilon at which the full sample reproduces the prediction a0 = a_Lambda / 2pi
ug_ = np.linspace(0.45, 0.75, 31)
a_of_u = np.array([best_rms(ALL, u)[0] for u in ug_])
for lab in ("Planck", "SH0ES"):
    t = PRED[lab]["planar"]
    u_match = np.interp(-t, -a_of_u, ug_)
    print(f"  full sample reproduces a0 = {t:.3e} ({lab}) at Upsilon_disk = {u_match:.2f}")
# chi2 (with intrinsic scatter) of the prediction against a free a0, at Upsilon 0.5 and 0.6
for u in (0.5, 0.6):
    for lab, sel in (("all", ALL), ("gas-rich", gas_rich)):
        si = sint_for(sel, u); b = best_chi(sel, u, si)
        gbu = gbar(u); kk = sel & good & (gbu > 0)
        r_ = np.log10(gobs[kk]) - np.log10(g_pred(gbu[kk], PRED["Planck"]["planar"]))
        print(f"  Ud={u} {lab:8s}: best a0 {b[0]:.3e}; prediction (Planck) dchi2 = {np.sum(r_**2/(sigv[kk]**2+si**2)) - b[1]:+.1f}")

# Figure 2: radial acceleration relation with the predicted a0
gb = gbar(0.5); k = good & (gb > 0)
lx, ly = np.log10(gb[k]), np.log10(gobs[k])
fig = plt.figure(figsize=(3.45, 3.9))
a1 = fig.add_axes([0.16, 0.36, 0.8, 0.6]); a2 = fig.add_axes([0.16, 0.1, 0.8, 0.24], sharex=a1)
a1.scatter(lx, ly, s=1.2, color=GREY, alpha=0.35, lw=0, rasterized=True)
edges = np.arange(-12.4, -8.4, 0.2); cen = (edges[1:] + edges[:-1]) / 2
med = [np.median(ly[(lx >= e0) & (lx < e1)]) if np.sum((lx >= e0) & (lx < e1)) > 10 else np.nan for e0, e1 in zip(edges[:-1], edges[1:])]
a1.plot(cen, med, "o", ms=3, color="k", label=r"binned median ($\Upsilon_{\rm disk}=0.5$)")
xx = np.linspace(-12.4, -8.4, 300)
a1.plot(xx, xx, ls=":", color=GREY, lw=0.8, label="no extra pull")
a1.plot(xx, np.log10(g_pred(10**xx, PRED["Planck"]["planar"])), color=BLUE, lw=1.2,
        label=fr"prediction $a_\Lambda/2\pi={PRED['Planck']['planar']*1e10:.2f}\times10^{{-10}}$")
a1.plot(xx, np.log10(g_pred(10**xx, a_rms)), color=ORANGE, ls="--", lw=1.0,
        label=fr"free fit $a_0={a_rms*1e10:.2f}\times10^{{-10}}$")
a1.set_ylabel(r"$\log_{10} g_{\rm obs}$ [m s$^{-2}$]"); a1.legend(frameon=False, loc="upper left", fontsize=6.8)
a1.set_xlim(-12.3, -8.6); a1.set_ylim(-11.6, -8.5); plt.setp(a1.get_xticklabels(), visible=False)
for u, col, ls in ((0.5, BLUE, "-"), (0.62, GREEN, "--")):
    gbu = gbar(u); kk = good & (gbu > 0); lxu = np.log10(gbu[kk])
    res = np.log10(gobs[kk]) - np.log10(g_pred(gbu[kk], PRED["Planck"]["planar"]))
    mres = [np.median(res[(lxu >= e0) & (lxu < e1)]) if np.sum((lxu >= e0) & (lxu < e1)) > 10 else np.nan for e0, e1 in zip(edges[:-1], edges[1:])]
    a2.plot(cen, mres, color=col, ls=ls, marker="o", ms=2.5, lw=1, label=fr"$\Upsilon_{{\rm disk}}={u}$")
a2.axhline(0, color="k", lw=0.6); a2.set_ylim(-0.25, 0.25); a2.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
a2.set_xlabel(r"$\log_{10} g_{\rm bar}$ [m s$^{-2}$]"); a2.set_ylabel("residual from\nprediction [dex]")
fig.savefig(os.path.join(FIGDIR, "fig_rar.pdf"), dpi=300); plt.close(fig)

# Figure 3: a0 estimates against the prediction, and the size test
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"wspace": 0.42})
a = ax[0]
lo_, hi_ = PRED["Planck"]["planar"] * 1e10, PRED["SH0ES"]["planar"] * 1e10
a.axhspan(lo_, hi_, color=BLUE, alpha=0.22, lw=0)
a.text(0.21, hi_ + 0.03, r"prediction $a_\Lambda/2\pi$ (Planck–SH0ES)", color=BLUE, fontsize=7, va="bottom", ha="left")
ua = np.array(list(ups_all)); va = np.array([ups_all[u] for u in ua]) * 1e10
a.plot(ua, va, "o-", color="k", ms=3.5, lw=0.9, label="all 164 galaxies")
ug = np.array(list(ups_gas)); vg = np.array([ups_gas[u] for u in ug]) * 1e10
a.plot(ug, vg, "s--", color=GREEN, ms=3.5, lw=0.9, label=f"{len(gas_names)} gas-rich galaxies")
a.errorbar([0.5], [b_me * 1e10], yerr=[[(b_me - b_lo) * 1e10], [(b_hi - b_me) * 1e10]], fmt="none", ecolor="k", capsize=2, lw=0.8)
a.errorbar([0.52], [g_me * 1e10], yerr=[[(g_me - g_lo) * 1e10], [(g_hi - g_me) * 1e10]], fmt="none", ecolor=GREEN, capsize=2, lw=0.8)
a.set_xlabel(r"stellar mass-to-light ratio $\Upsilon_{\rm disk}$ [3.6 $\mu$m]"); a.set_ylabel(r"best-fit $a_0$ [$10^{-10}$ m s$^{-2}$]")
a.set_xlim(0.2, 0.8); a.set_ylim(0.5, 2.0); a.legend(frameon=False, loc="upper right")
a = ax[1]
a.errorbar(Rd, a0g * 1e10, fmt="o", ms=2.5, color=GREY, alpha=0.7, lw=0)
a.axhspan(lo_, hi_, color=BLUE, alpha=0.22, lw=0)
bins_r = np.geomspace(Rd.min(), Rd.max(), 7)
bc = np.sqrt(bins_r[1:] * bins_r[:-1]); bm = [np.median(a0g[(Rd >= b0) & (Rd < b1)]) * 1e10 for b0, b1 in zip(bins_r[:-1], bins_r[1:])]
a.plot(bc, bm, "s", color="k", ms=4, label="median per bin")
a.set_xscale("log"); a.set_yscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
a.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 4])); a.yaxis.set_major_formatter(FixedFormatter(["0.3", "0.5", "1", "2", "4"])); a.yaxis.set_minor_locator(NullLocator())
a.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10])); a.xaxis.set_major_formatter(FixedFormatter(["0.3", "1", "3", "10"])); a.xaxis.set_minor_locator(NullLocator())
a.set_xlabel(r"disk scale length $R_d$ [kpc]"); a.set_ylabel(r"per-galaxy $a_0$ [$10^{-10}$ m s$^{-2}$]")
a.text(0.04, 0.05, fr"Spearman $\rho={rho_d:+.2f}$ ($p={p_d:.2f}$), $N={len(rows)}$", transform=a.transAxes, fontsize=7.5)
a.legend(frameon=False, loc="upper right")
fig.savefig(os.path.join(FIGDIR, "fig_a0.pdf")); plt.close(fig)
print(f"\nD. figures written to {FIGDIR}")

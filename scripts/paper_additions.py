"""
Additional tests for the paper.
  1. The nested ladder of rings, r^3 = G m / S, from the Moon to the Local Group
  2. Does a0 change with redshift?  41 star-forming disks at z = 0.65-2.45
     (Price et al. 2021, ApJ 922, 143, Table 3: z, R_e,disk, log M_bar, f_DM(R_e), 1D MAP values).
     If the local RAR holds at R_e:  f_DM = exp(-sqrt(g_bar/a0))  ->  a0 = g_bar / (ln f_DM)^2.
     Compare with a constant a0 and with a0 proportional to H(z).
  3. Robustness of the SPARC results: per-galaxy a0 against distance, surface brightness,
     rotation speed and gas fraction; gas-rich threshold; the RAR transition function.
Usage: python3 paper_additions.py <SPARC Rotmod_LTG folder>
"""
import sys, os, glob
import numpy as np
from scipy.special import i0, i1, k0, k1
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_a0 import load, g_pred

G = 6.674e-11; MSUN = 1.989e30; KPC = 3.0857e19; PC = 3.0857e16; C = 2.998e8; MPC = 3.0857e22
AU = 1.496e11

# ------------------------------------------------------------- 1. ladder
print("1. The ladder of rings, r^3 = G m / S")
H0, OL, OM = 67.4, 0.685, 0.315
H0s = H0 * 1e3 / MPC
S_L = H0s**2 * OL                                     # Lambda c^2 / 3
rows = [
    ("Moon in Earth's ring", 7.342e22, 3 * G * 5.972e24 / 3.844e8**3, "point-mass host, D = 384,400 km"),
    ("Earth in the Sun's ring", 5.972e24, 3 * G * MSUN / AU**3, "point-mass host, D = 1 au"),
    ("Sun in the Milky Way", MSUN, 2 * (2.3e5)**2 / (8.2 * KPC)**2, "flat rotation curve, V = 230 km/s, D = 8.2 kpc"),
]
for lab, m, S, note in rows:
    r = (G * m / S) ** (1 / 3)
    print(f"  {lab:26s} r = {r/1e3:12.4g} km = {r/PC:.3g} pc   ({note})")
print("  Milky Way in the Local Group: 269 kpc, M31: 323 kpc (Roche lobes, paper_analysis.py)")
for M in (1.9e12, 3e12, 5e12):
    print(f"  Local Group in emptiness (M = {M:.1e} Msun): R_Lambda = {(G*M*MSUN/S_L)**(1/3)/MPC:.2f} Mpc")

# ------------------------------------------------------------- 2. high redshift
print("\n2. a0 at high redshift (Price et al. 2021, 41 galaxies)")
P21 = [  # z, Re_disk [kpc], log Mbar, fDM(Re)
 (0.658,3.93,10.93,0.22),(0.669,8.68,10.80,0.69),(0.732,5.90,10.68,0.11),(0.748,5.49,11.11,0.24),(0.760,5.27,10.25,0.48),
 (0.809,7.57,10.41,0.45),(0.824,6.76,9.98,0.75),(0.831,9.40,10.71,0.62),(0.854,7.08,10.97,0.53),(0.914,9.56,10.54,0.58),
 (0.922,6.65,10.67,0.42),(0.990,4.71,10.13,0.76),(1.031,8.20,10.87,0.63),(1.032,9.22,11.12,0.49),(1.120,10.17,11.09,0.28),
 (1.197,4.48,11.12,0.08),(1.232,6.70,11.43,0.18),(1.362,4.17,10.94,0.47),(1.400,5.94,11.14,0.35),(1.446,3.28,10.60,0.05),
 (1.500,6.83,11.13,0.38),(1.530,7.80,11.25,0.34),(1.614,5.05,10.82,0.38),(1.615,7.35,11.45,0.30),(2.036,7.00,10.92,0.46),
 (2.154,6.03,10.46,0.52),(2.172,7.98,10.61,0.71),(2.180,6.81,11.05,0.59),(2.182,5.39,10.66,0.61),(2.187,8.80,10.91,0.25),
 (2.196,4.84,11.06,0.06),(2.210,6.02,11.06,0.45),(2.225,7.70,10.89,0.76),(2.236,4.88,10.69,0.13),(2.242,6.18,10.98,0.29),
 (2.258,6.58,11.05,0.61),(2.308,7.08,10.85,0.43),(2.383,6.40,11.13,0.24),(2.387,6.05,11.52,0.12),(2.431,5.17,10.78,0.43),
 (2.451,5.17,10.43,0.43)]
z, Re, lM, f = map(np.array, zip(*P21))
# baryonic acceleration at Re: half of M_bar inside Re, two geometries bracket the answer
y = 1.678 / 2                                          # Re = 1.678 Rd, y = R / 2Rd
disk_factor = 2 * (2 * y)**2 * (i0(y) * k0(y) - i1(y) * k1(y)) / 1.678 * 1.678  # V^2 R / (G M_tot) for a thin exponential disk at Re
# V^2 = (G M / Rd) 2 y^2 [I0K0 - I1K1]  ->  g = V^2/R
def gbar_disk(M, Re_kpc):
    Rd = Re_kpc / 1.678 * KPC
    V2 = G * M / Rd * 2 * y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
    return V2 / (Re_kpc * KPC)
def gbar_sphere(M, Re_kpc):
    return G * 0.5 * M / (Re_kpc * KPC)**2
M = 10**lM * MSUN
H = lambda zz: np.sqrt(OM * (1 + zz)**3 + OL)
A0_LOCAL = 1.14e-10
for lab, gfun in (("thin exponential disk", gbar_disk), ("spherical, half mass inside Re", gbar_sphere)):
    gb = gfun(M, Re)
    a0 = gb / np.log(f)**2
    print(f"  geometry: {lab}")
    print(f"    g_bar(Re): median {np.median(gb):.2e} m/s^2 ({np.median(gb)/A0_LOCAL:.1f} a0_local)")
    for zlo, zhi in ((0.6, 1.2), (1.2, 2.0), (2.0, 2.5), (0.6, 2.5)):
        k = (z >= zlo) & (z < zhi)
        rng = np.random.default_rng(3)
        bs = [np.median(rng.choice(a0[k], k.sum())) for _ in range(4000)]
        lo, hi = np.percentile(bs, [16, 84]); med = np.median(a0[k])
        zz = np.median(z[k])
        print(f"    z {zlo}-{zhi} (n={k.sum():2d}, median z {zz:.2f}): a0 = {med:.2e} (+{hi-med:.1e}/-{med-lo:.1e});  "
              f"constant predicts {A0_LOCAL:.2e}, H(z) scaling predicts {A0_LOCAL*H(zz):.2e}")
    rho, p = spearmanr(z, a0)
    slope = np.polyfit(np.log10(H(z)), np.log10(a0), 1)[0]
    print(f"    a0 vs z: Spearman {rho:+.2f} (p = {p:.2f});  fitted d log a0 / d log H = {slope:+.2f}  (constant: 0, H scaling: 1)")
    # predicted fDM for the two hypotheses vs observed
    for hyp, a in (("constant", A0_LOCAL * np.ones_like(z)), ("H(z) scaling", A0_LOCAL * H(z))):
        fp = np.exp(-np.sqrt(gb / a))
        print(f"    {hyp:13s}: predicted median f_DM {np.median(fp):.2f}; observed {np.median(f):.2f}; "
              f"rms(obs - pred) {np.sqrt(np.mean((f-fp)**2)):.2f}; mean(obs - pred) {np.mean(f-fp):+.2f}")

# ------------------------------------------------------------- 3. robustness
print("\n3. Robustness of the SPARC fits")
folder = sys.argv[1]
names, arr = load(folder)
dist = {os.path.basename(p)[:-11]: float(open(p).readline().split('=')[1].split()[0]) for p in glob.glob(folder + '/*_rotmod.dat')}
sb0 = {}
for p in glob.glob(folder + '/*_rotmod.dat'):
    r0 = [l.split() for l in open(p) if not l.startswith('#') and l.strip()][0]
    sb0[os.path.basename(p)[:-11]] = float(r0[6])
R, Vobs, eV, Vgas, Vdisk, Vbul = arr.T
K = 1e6 / 3.0856776e19; sq = lambda v: v * np.abs(v)
gobs = Vobs**2 / R * K; fe = eV / Vobs; sig = np.maximum(2 * fe / np.log(10), 0.02); good = fe < 0.1
gbar = lambda ud: (sq(Vgas) + ud * sq(Vdisk) + 1.4 * ud * sq(Vbul)) / R * K
GRID = np.geomspace(0.3e-10, 4e-10, 1200)
def best(sel, ud=0.5, form="exp"):
    gb = gbar(ud); k = sel & good & (gb > 0); lo = np.log10(gobs[k])
    c = [np.sum(((lo - np.log10(g_pred(gb[k], a, form))) / sig[k])**2) for a in GRID]
    return GRID[int(np.argmin(c))]
gb = gbar(0.5); res = []
for g in sorted(set(names)):
    kk = names == g; k = kk & good & (gb > 0)
    if k.sum() < 8: continue
    lo = np.log10(gobs[k]); c = np.array([np.sum(((lo - np.log10(g_pred(gb[k], a))) / sig[k])**2) for a in GRID]); i = c.argmin()
    if i in (0, len(GRID) - 1): continue
    gas = np.mean(np.maximum(sq(Vgas[kk]), 0) / np.maximum(gb[kk] * R[kk] / K, 1e-9))
    res.append((g, GRID[i], dist[g], sb0[g], Vobs[kk].max(), gas))
a0g = np.array([r[1] for r in res])
for j, lab in ((2, "distance"), (3, "central disk surface brightness"), (4, "maximum rotation speed"), (5, "gas fraction")):
    x = np.array([r[j] for r in res]); m = np.isfinite(x) & (x > 0)
    rho, p = spearmanr(x[m], a0g[m])
    print(f"  per-galaxy a0 vs {lab:31s}: Spearman {rho:+.2f} (p = {p:.2f}, N = {m.sum()})")
def gasrich(th):
    sel = np.zeros(len(R), bool); n = 0
    for g in sorted(set(names)):
        k = (names == g) & (gb > 0) & (gb < 3e-10)
        if k.sum() < 4: continue
        share = np.mean(np.maximum(sq(Vgas[k]), 0) / np.maximum(sq(Vgas[k]) + 0.5 * sq(Vdisk[k]) + 0.7 * sq(Vbul[k]), 1e-9))
        if share > th: sel |= names == g; n += 1
    return sel, n
for th in (0.5, 0.6, 0.7):
    s, n = gasrich(th); print(f"  gas-rich threshold {th:.0%}: {n} galaxies, a0 = {best(s):.3e}")
s, _ = gasrich(0.6); allg = np.ones(len(R), bool)
print(f"  transition function: all galaxies exp {best(allg):.3e} / simple {best(allg, form='simple'):.3e};  "
      f"gas-rich exp {best(s):.3e} / simple {best(s, form='simple'):.3e}")

# ------------------------------------------------------------- 2b. calibration-free test within the high-z sample
print("\n2b. Within-sample change of a0 (cancels the overall calibration offset)")
rng = np.random.default_rng(11)
for lab, gfun in (("thin exponential disk", gbar_disk), ("spherical", gbar_sphere)):
    a0 = gfun(M, Re) / np.log(f)**2
    lo_k, hi_k = z < 1.2, z >= 2.0
    ratio = np.median(a0[hi_k]) / np.median(a0[lo_k])
    pred = H(np.median(z[hi_k])) / H(np.median(z[lo_k]))
    bs_r, bs_s = [], []
    for _ in range(4000):
        i = rng.integers(0, len(z), len(z))
        a, zz = a0[i], z[i]
        if (zz < 1.2).sum() > 2 and (zz >= 2).sum() > 2:
            bs_r.append(np.median(a[zz >= 2]) / np.median(a[zz < 1.2]))
        bs_s.append(np.polyfit(np.log10(H(zz)), np.log10(a), 1)[0])
    r16, r84 = np.percentile(bs_r, [16, 84]); r97 = np.percentile(bs_r, 97.5)
    s16, s84 = np.percentile(bs_s, [16, 84]); s97 = np.percentile(bs_s, 97.5)
    print(f"  {lab}: a0(z>2)/a0(z<1.2) = {ratio:.2f} (68%: {r16:.2f}-{r84:.2f}; 97.5% upper {r97:.2f}); H(z) scaling predicts {pred:.2f}, constant 1.00")
    print(f"     slope d log a0 / d log H = {np.median(bs_s):+.2f} (68%: {s16:+.2f} to {s84:+.2f}; 97.5% upper {s97:+.2f})")

# ------------------------------------------------------------- 4. does a galaxy's own shape set a0?
print("\n4. Visible shape: bulge-rich against pure-disc galaxies")
bf = {}
for g in sorted(set(names)):
    kk = names == g
    tot = np.max((sq(Vgas[kk]) + 0.5 * sq(Vdisk[kk]) + 0.7 * sq(Vbul[kk])) * R[kk])
    bf[g] = np.max(0.7 * sq(Vbul[kk]) * R[kk]) / tot if tot > 0 else 0.0
from scipy.stats import mannwhitneyu
bulgy = np.array([bf[n] > 0.1 for n in names]); pdisc = np.array([bf[n] == 0 for n in names])
print(f"  pooled fit: bulge-rich ({len({n for n in names if bf[n] > 0.1})} galaxies) a0 = {best(bulgy):.3e};  "
      f"pure discs ({len({n for n in names if bf[n] == 0})}) a0 = {best(pdisc):.3e}")
xb = np.array([bf[r[0]] for r in res]); rho, p = spearmanr(xb, a0g)
yb, yd = a0g[xb > 0.1], a0g[xb == 0]
print(f"  per-galaxy: Spearman with bulge fraction {rho:+.2f} (p = {p:.2f}); medians bulge-rich {np.median(yb):.2e} (n={len(yb)}), "
      f"pure disc {np.median(yd):.2e} (n={len(yd)}), Mann-Whitney p = {mannwhitneyu(yb, yd).pvalue:.2f}")

# ------------------------------------------------------------- figure: a0 against redshift
if len(sys.argv) > 2:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "legend.fontsize": 7.5,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    a0z = gbar_disk(M, Re) / np.log(f)**2
    fig, a = plt.subplots(figsize=(3.45, 2.6))
    a.scatter(z, a0z * 1e10, s=10, color="#2c5d8a", lw=0, label="41 disks (Price et al. 2021)")
    for zlo, zhi in ((0.6, 1.2), (1.2, 2.0), (2.0, 2.5)):
        kk = (z >= zlo) & (z < zhi); a.plot(np.median(z[kk]), np.median(a0z[kk]) * 1e10, "s", color="k", ms=6)
    zz = np.linspace(0, 2.6, 100)
    a.plot(zz, np.full_like(zz, A0_LOCAL * 1e10), color="#d9822b", lw=1.1, label=r"constant $a_0$")
    a.plot(zz, A0_LOCAL * H(zz) * 1e10, color="#b23b3b", ls="--", lw=1.1, label=r"$a_0\propto H(z)$")
    a.set_yscale("log"); a.set_xlim(0, 2.6); a.set_ylim(0.2, 30)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    a.yaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30])); a.yaxis.set_major_formatter(FixedFormatter(["0.3", "1", "3", "10", "30"])); a.yaxis.set_minor_locator(NullLocator())
    a.set_xlabel("redshift $z$"); a.set_ylabel(r"$a_0$ implied at $R_e$ [$10^{-10}$ m s$^{-2}$]")
    a.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(sys.argv[2], "fig_highz.pdf")); plt.close(fig)
    print("  figure written")

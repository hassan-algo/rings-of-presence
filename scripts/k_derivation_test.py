"""
k = pi (1 + x), x = (average presence inside the ring) / (presence at the ring) - 1.
  Version A (presence of a point, 3-D ball): x = 1/2 for every object -> k = 3 pi / 2.
  Version B (presence of the actual mass layout): x depends on how spread out the mass is
            relative to the a0 ring (exponential sphere with the disk's scale length).
Tests:
  1. Gas-rich galaxies (star weights matter least): best a0 for star weights 0.25 / 0.5 / 0.75.
  2. Version A vs B: total misfit with each, and whether galaxy-by-galaxy a0 follows B's prediction.
"""
import sys, numpy as np
from scipy.integrate import quad
from scipy.stats import spearmanr
from fit_a0 import load, g_pred
KMS2_PER_KPC = 1e6 / 3.0856776e19
G = 4.30091e-6
C = 299792458.0; MPC = 3.0856776e22
BASE = {"Planck": C * 67.4e3 / MPC * np.sqrt(0.685), "SH0ES": C * 73.0e3 / MPC * np.sqrt(0.70)}

# x(r/a) for an exponential sphere (B)
rho = lambda t: np.exp(-t) / (8 * np.pi)
Menc = lambda s: quad(lambda t: 4 * np.pi * t * t * rho(t), 0, s)[0]
Pot = lambda s: Menc(s) / s + quad(lambda t: 4 * np.pi * rho(t) * t, s, np.inf)[0]
def ratio(r):
    return quad(lambda s: Pot(s) * 4 * np.pi * s * s, 0, r, limit=200)[0] / (4 / 3 * np.pi * r**3) / Pot(r)
grid_u = np.geomspace(0.05, 200, 80); grid_x = np.array([ratio(u) - 1 for u in grid_u])
xB = lambda u: np.interp(np.log(u), np.log(grid_u), grid_x)

names, arr = load(sys.argv[1])
R, Vobs, eV, Vgas, Vdisk, Vbul = arr.T
sq = lambda v: v * np.abs(v)
def gbar(ud, ub=None):
    ub = 1.4 * ud if ub is None else ub
    return (sq(Vgas) + ud * sq(Vdisk) + ub * sq(Vbul)) / R * KMS2_PER_KPC
gobs = Vobs**2 / R * KMS2_PER_KPC
fe = eV / Vobs
sig = np.maximum(2 * fe / np.log(10), 0.02)
good = fe < 0.1

def best_a0(sel, ud):
    gb = gbar(ud); k = sel & good & (gb > 0)
    grid = np.geomspace(0.3e-10, 4e-10, 1500)
    chi = [np.sum(((np.log10(gobs[k]) - np.log10(g_pred(gb[k], a))) / sig[k]) ** 2) for a in grid]
    return grid[int(np.argmin(chi))]

# 1. gas-rich galaxies: gas share of the baryonic pull where g_bar < 3e-10 (star weight 0.5)
gb05 = gbar(0.5)
gas_rich = np.zeros(len(R), bool); ng = 0
for g in set(names):
    k = (names == g) & (gb05 > 0) & (gb05 < 3e-10)
    if k.sum() < 4: continue
    share = np.mean(np.maximum(sq(Vgas[k]), 0) / np.maximum(sq(Vgas[k]) + 0.5 * sq(Vdisk[k]) + 0.7 * sq(Vbul[k]), 1e-9))
    if share > 0.6:
        gas_rich |= (names == g); ng += 1
print(f"1. Gas-rich galaxies (gas > 60% of the ordinary pull in the low-pull region): {ng}")
for ud in (0.25, 0.5, 0.75):
    print(f"   star weight {ud}: best a0 = {best_a0(gas_rich, ud):.3e}   (all galaxies: {best_a0(np.ones(len(R), bool), ud):.3e})")
for lab, b in BASE.items():
    print(f"   predictions with {lab}: k=2pi -> {b/(2*np.pi):.3e};  k=3pi/2 -> {b/(1.5*np.pi):.3e};  k=4pi/3 -> {b/(4*np.pi/3):.3e}")

# 2. version A vs B, star weight 0.5, Planck
base = BASE["Planck"]; aA = base / (1.5 * np.pi)
chiA = chiB = 0; preds = []; fits = []
grid = np.geomspace(0.3e-10, 6e-10, 800)
for g in sorted(set(names)):
    kk = names == g
    k = kk & good & (gb05 > 0)
    if k.sum() < 8: continue
    Rd = R[kk][np.argmax(np.abs(Vdisk[kk]))] / 2.2
    Mb = max(sq(Vgas[kk]) + 0.5 * sq(Vdisk[kk]) + 0.7 * sq(Vbul[kk]))  # placeholder
    Mtot = (sq(Vgas[kk]) + 0.5 * sq(Vdisk[kk]) + 0.7 * sq(Vbul[kk]))[-1] * R[kk][-1] / G   # ordinary mass inside last point
    a = aA
    for _ in range(5):
        ra = np.sqrt(G * Mtot / (a / KMS2_PER_KPC))   # kpc
        a = base / (np.pi * (1 + xB(max(ra / Rd, 0.05))))
    lo = np.log10(gobs[k]); s = sig[k]
    chiA += np.sum(((lo - np.log10(g_pred(gb05[k], aA))) / s) ** 2)
    chiB += np.sum(((lo - np.log10(g_pred(gb05[k], a))) / s) ** 2)
    c = [np.sum(((lo - np.log10(g_pred(gb05[k], t))) / s) ** 2) for t in grid]
    i = int(np.argmin(c))
    if 0 < i < len(grid) - 1:
        preds.append(a); fits.append(grid[i])
preds, fits = np.array(preds), np.array(fits)
print(f"\n2. Version A (x = 1/2 everywhere, a0 = {aA:.3e}) vs version B (x from each galaxy's mass layout)")
print(f"   B's predicted a0 ranges {np.percentile(preds,5):.2e} to {np.percentile(preds,95):.2e} (median {np.median(preds):.2e})")
print(f"   total misfit (chi2): A = {chiA:.0f}, B = {chiB:.0f}  (lower is better)")
rho_s, p = spearmanr(preds, fits)
print(f"   do galaxies' own a0 follow B's prediction? rank correlation {rho_s:+.2f} (p = {p:.2f}, n = {len(fits)})")

# 3. how each candidate fits, and the gas-rich uncertainty
print("\n3. Misfit (chi2) for each candidate, star weight 0.5")
gb = gbar(0.5)
def chi(sel, a):
    k = sel & good & (gb > 0)
    return np.sum(((np.log10(gobs[k]) - np.log10(g_pred(gb[k], a))) / sig[k]) ** 2)
allg = np.ones(len(R), bool)
for lab, a in (("k = 2pi (flat ring), Planck", BASE["Planck"] / (2 * np.pi)), ("k = 2pi, SH0ES", BASE["SH0ES"] / (2 * np.pi)),
               ("k = 3pi/2 (ball), Planck", BASE["Planck"] / (1.5 * np.pi)), ("k = 3pi/2, SH0ES", BASE["SH0ES"] / (1.5 * np.pi))):
    print(f"   {lab:30s} a0 = {a:.2e}: all galaxies {chi(allg, a):8.0f}   gas-rich {chi(gas_rich, a):6.0f}")
gal = sorted(set(names[gas_rich])); rng = np.random.default_rng(4); bs = []
for _ in range(400):
    pick = rng.choice(gal, len(gal))
    sel = np.zeros(len(R), bool)
    w = np.zeros(len(R))
    for p in pick: w += (names == p)
    k = (w > 0) & good & (gb > 0)
    grid = np.geomspace(0.3e-10, 3e-10, 400)
    c = [np.sum(w[k] * ((np.log10(gobs[k]) - np.log10(g_pred(gb[k], a))) / sig[k]) ** 2) for a in grid]
    bs.append(grid[int(np.argmin(c))])
lo, me, hi = np.percentile(bs, [16, 50, 84])
print(f"   gas-rich a0 with galaxy-to-galaxy uncertainty: {me:.2e} (+{hi-me:.2e}/-{me-lo:.2e})")
print("   gas-rich galaxies:", ", ".join(gal))

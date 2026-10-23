"""
Test k = pi (1 + x).
1. What x do the galaxies need?  a0 = c^2 sqrt(Lambda/3) / k = c H0 sqrt(Omega_L) / k
2. If x depends on each object's own area, a0 must change from galaxy to galaxy with size.
   Fit a0 separately for each SPARC galaxy (fixed star weights 0.5 disk / 0.7 bulge) and
   compare with size (disk scale length from where the disk's own speed peaks, R_peak/2.2,
   and the area pi Rd^2).
"""
import sys, numpy as np
from scipy.stats import spearmanr
from fit_a0 import load, accelerations, g_pred
C = 299792458.0; MPC = 3.0856776e22
for lab, H0, OL in (("Planck", 67.4, 0.685), ("SH0ES", 73.0, 0.70)):
    base = C * H0 * 1e3 / MPC * np.sqrt(OL)
    print(f"{lab}: c H0 sqrt(OmegaL) = {base:.3e}")
    for a0 in (0.91e-10, 1.13e-10, 1.45e-10):
        k = base / a0
        print(f"   a0 = {a0:.2e}: k = {k:.2f}, x = k/pi - 1 = {k/np.pi-1:.2f}")
    print(f"   small x (x = 0): a0 = {base/np.pi:.2e}")

names, arr = load(sys.argv[1])
gobs, gbar, fe = accelerations(arr)
R, Vobs, eV, Vgas, Vdisk, Vbul = arr.T
grid = np.geomspace(0.2e-10, 8e-10, 600)
rows = []
for g in sorted(set(names)):
    k = (names == g) & (gbar > 0) & (fe < 0.1)
    if k.sum() < 8: continue
    lo, lb = np.log10(gobs[k]), np.log10(gbar[k])
    sig = np.maximum(2 * fe[k] / np.log(10), 0.02)
    chi = np.array([np.sum(((lo - np.log10(g_pred(gbar[k], a))) / sig) ** 2) for a in grid])
    i = chi.argmin()
    if i in (0, len(grid) - 1): continue
    ok = grid[chi <= chi[i] + 1]
    kk = names == g
    Rpk = R[kk][np.argmax(np.abs(Vdisk[kk]))]
    rows.append((g, grid[i], np.log10(ok.max() / ok.min()) / 2, Rpk / 2.2, R[kk].max()))
a0g = np.array([r[1] for r in rows]); err = np.array([r[2] for r in rows])
Rd = np.array([r[3] for r in rows]); Rmax = np.array([r[4] for r in rows])
print(f"\n{len(rows)} galaxies fitted one by one")
print(f"  median a0 = {np.median(a0g):.2e}; spread {np.std(np.log10(a0g)):.2f} dex; typical fit error {np.median(err):.2f} dex")
for lab, s in (("disk scale length", Rd), ("disk area pi Rd^2", np.pi * Rd**2), ("outermost radius", Rmax)):
    rho, p = spearmanr(s, a0g)
    print(f"  a0 against {lab:20s}: rank correlation {rho:+.2f} (p = {p:.3f})")
# split by size
med = np.median(Rd)
sm, bg = a0g[Rd < med], a0g[Rd >= med]
print(f"  smaller half (Rd < {med:.1f} kpc): median a0 {np.median(sm):.2e};  larger half: {np.median(bg):.2e}")
xg = (C * 67.4e3 / MPC * np.sqrt(0.685)) / (np.pi * a0g) - 1
print(f"  x each galaxy needs (Planck): median {np.median(xg):.2f}, middle 68% {np.percentile(xg,16):.2f} to {np.percentile(xg,84):.2f}")
# what would an area-based x predict? x = (ring area)/(object area) with a fixed ring width w: x = 2w/Rd + (w/Rd)^2
print("  if x = 2w/Rd + (w/Rd)^2 for a fixed ring width w, a0 would RISE with size; for x fixed, a0 is flat.")

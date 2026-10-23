"""
Test 2: does each galaxy's own outside pull (from its neighbours) show up in its outskirts?

Two versions of rule 13 for a rotating disk galaxy:
  (a) 1-D form:      g = (g_bar + e a0) nu((g_bar + e a0)/a0) - e a0 nu(e),  nu(y) = 1/(1 - exp(-sqrt y))
  (b) Chae et al. 2020, eq. 6 (calibrated on full MOND calculations for disks):
        g = g_bar * nu_e(z),  z = g_bar / a0
        nu_e(z) = 1/2 - A/z + sqrt((1/2 - A/z)^2 + B/z),  A = e(1 + e/2)/(1 + e),  B = 1 + e
  e = outside pull / a0.  Einstein + dark matter: e has no effect.

Per galaxy: find the e that fits its rotation curve best, and compare with the outside pull
estimated from its neighbours (Chae et al. 2020, erratum Table 2) for their showcase galaxies.

Usage: python3 galaxy_outside_pull.py <folder with SPARC *_rotmod.dat files>
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_a0 import load, accelerations

A0 = 1.13e-10


def nu(y):
    return 1.0 / (-np.expm1(-np.sqrt(np.maximum(y, 1e-30))))


def g_1d(gbar, e, a0=A0):
    tot = gbar + e * a0
    return tot * nu(tot / a0) - e * a0 * nu(e)


def g_chae(gbar, e, a0=1.2e-10):
    z = gbar / a0
    A = e * (1 + e / 2) / (1 + e)
    B = 1 + e
    return gbar * (0.5 - A / z + np.sqrt((0.5 - A / z) ** 2 + B / z))


def rms(gobs, pred):
    return np.sqrt(np.mean((np.log10(gobs) - np.log10(pred)) ** 2))


# neighbour-based outside pulls (Chae et al. 2020, erratum Table 2), units of a0
SHOWCASE = {"NGC5033": (0.050, "strong neighbours"), "NGC5055": (0.040, "strong neighbours"),
            "NGC1090": (0.011, "isolated"), "NGC6674": (0.011, "isolated")}

if __name__ == "__main__":
    names, arr = load(sys.argv[1])
    gobs, gbar, fe = accelerations(arr)
    k = (fe < 0.10) & (gbar > 0)
    names, gobs, gbar = names[k], gobs[k], gbar[k]
    gal = sorted(set(names))
    es = np.concatenate([[0.0], np.geomspace(1e-3, 0.3, 120)])

    print("=== All galaxies, one shared outside pull (Chae's average from neighbours: e = 0.033) ===")
    for label, f in (("1-D form", g_1d), ("Chae form", g_chae)):
        r0, r33 = rms(gobs, f(gbar, 0.0)), rms(gobs, f(gbar, 0.033))
        print(f"  {label:10s}: no outside pull {r0:.4f} dex   e = 0.033 for all {r33:.4f} dex")

    print("\n=== Each galaxy fitted with its own outside pull (Chae form) ===")
    best, gain = {}, {}
    for g in gal:
        ii = names == g
        if ii.sum() < 6:
            continue
        m = np.array([rms(gobs[ii], g_chae(gbar[ii], e)) for e in es])
        best[g] = es[m.argmin()]
        gain[g] = m[0] - m.min()
    b = np.array(list(best.values()))
    print(f"  {len(b)} galaxies with >= 6 points")
    print(f"  prefer no outside pull (e < 0.002): {np.mean(b < 0.002) * 100:.0f}%")
    print(f"  prefer 0.002-0.03: {np.mean((b >= 0.002) & (b < 0.03)) * 100:.0f}%   "
          f"0.03-0.1: {np.mean((b >= 0.03) & (b < 0.1)) * 100:.0f}%   above 0.1: {np.mean(b >= 0.1) * 100:.0f}%")
    print(f"  median fitted e {np.median(b):.3f}  (neighbour-based average: 0.033)")

    print("\n  Showcase galaxies (outside pull from neighbours vs fitted from the rotation curve):")
    for g, (e_env, kind) in SHOWCASE.items():
        if g not in best:
            print(f"    {g}: not in sample after cuts")
            continue
        ii = names == g
        r_env = rms(gobs[ii], g_chae(gbar[ii], e_env))
        r_0 = rms(gobs[ii], g_chae(gbar[ii], 0.0))
        print(f"    {g:8s} ({kind:17s}) neighbours e = {e_env:.3f}   fitted e = {best[g]:.3f}   "
              f"fit: no outside pull {r_0:.3f} dex -> with neighbours' pull {r_env:.3f} dex")

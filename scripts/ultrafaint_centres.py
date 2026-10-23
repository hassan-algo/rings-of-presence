"""
Heat vs bumping sand: are the centres of satellites with almost no stars still dense?

If only heat from young stars smooths dark matter (rule 10e), galaxies with too few stars
(ultra-faint satellites, M* < ~1e5 Msun) must keep the dense centre ("cusp") of untouched
dark matter. Bumping sand (rule 10c) can smooth them anyway, or make them collapse denser.

Untouched-dark-matter expectation (cold dark matter, NFW clump):
    halo mass before falling into the Milky Way: M200 = 1e8 - 3e9 Msun (range for ultra-faints)
    concentration: log10 c = 1.025 - 0.097 log10(M200 / (1e12/h)),  h = 0.7  (Dutton & Maccio 2014)
    mean density inside the stars' 3-D half-light radius r:  rho = M_NFW(<r) / (4/3 pi r^3)
    tides from the Milky Way can lower it by up to ~3x while keeping the dense centre
Measured: rho = 3 sigma^2 r / G / (4/3 pi r^3)   (same as before)

Verdict per galaxy:
    below band/3  -> smooth-centre candidate (needs something beyond star heating)
    within        -> consistent with an untouched dense centre
    above band    -> denser than any untouched clump (collapsed centre, or speeds inflated by binary stars)
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite_predictions as sp

G = 4.30091e-6            # kpc (km/s)^2 / Msun
H = 0.7
RHO_CRIT = 277.5 * H * H  # Msun / kpc^3


def nfw_mean_density(M200, r):
    c = 10 ** (1.025 - 0.097 * math.log10(M200 / (1e12 / H)))
    r200 = (3 * M200 / (4 * math.pi * 200 * RHO_CRIT)) ** (1 / 3)
    rs = r200 / c
    mu = lambda x: math.log(1 + x) - x / (1 + x)
    Menc = M200 * mu(r / rs) / mu(c)
    return Menc / (4 / 3 * math.pi * r**3)


if __name__ == "__main__":
    folder = sys.argv[1]
    rows = []
    for g in sp.load(folder):
        r = g["raw"]
        if not r["vlos_sigma"] or r["vlos_sigma_ul"]:
            continue
        s = float(r["vlos_sigma"])
        r12 = 4 / 3 * g["Re"] / 1000
        rho_obs = 3 * s * s * r12 / G / (4 / 3 * math.pi * r12**3)
        lo = nfw_mean_density(1e8, r12)
        hi = nfw_mean_density(3e9, r12)
        Mstar = 2 * 10 ** (-0.4 * (g["MV"] - 4.83))
        verdict = ("smooth-centre candidate" if rho_obs < lo / 3 else
                   ("denser than untouched" if rho_obs > hi else "dense centre OK"))
        rows.append(dict(name=g["name"], host=g["host"], MV=g["MV"], Mstar=Mstar, r=r12 * 1000, s=s,
                         rho=rho_obs / 1e9, lo=lo / 1e9, hi=hi / 1e9, v=verdict))

    for label, sel in (("ULTRA-FAINT (too few stars to heat, M* < ~1e5)", [o for o in rows if o["Mstar"] < 1.2e5]),
                       ("BRIGHTER (enough stars to heat)", [o for o in rows if o["Mstar"] >= 1.2e5])):
        print(f"\n=== {label}: {len(sel)} satellites ===")
        print(f"  {'satellite':18s} {'stars':>9s} {'r_half pc':>9s} {'measured':>9s} {'untouched range':>17s}  verdict")
        for o in sorted(sel, key=lambda o: o["rho"] / o["lo"]):
            print(f"  {o['name'][:18]:18s} {o['Mstar']:9.1e} {o['r']:9.0f} {o['rho']:9.3f} "
                  f"{o['lo']:7.3f} - {o['hi']:6.3f}  {o['v']}")
        n = len(sel)
        for v in ("smooth-centre candidate", "dense centre OK", "denser than untouched"):
            k = sum(o["v"] == v for o in sel)
            print(f"  {v:26s} {k:3d} / {n}  ({100 * k / n:.0f}%)")
    print("\n(densities in Suns per cubic parsec; 'untouched range' = NFW clump of 1e8 to 3e9 Suns)")

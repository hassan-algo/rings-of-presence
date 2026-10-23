"""
Round 2, test B: nested control rings.  r^3 = G m / S, S = 3GM/D^3 (host tide) or Lambda c^2/3 (emptiness).
B1. Planets: Hill ring R_H = a (m/3M_sun)^(1/3); outermost known moon (JPL SSD / discovery papers) must sit inside,
    and inside the stable part (about 0.5 R_H prograde, 0.7 R_H retrograde; Hamilton & Krivov 1997).
B2. Milky Way and M31 satellites vs their rings (Roche share of the Local Group, Eggleton 1983 formula).
B3. Local Group balance ring measured from galaxy velocities (Local Volume Database, Pace 2024) vs
    the largest ring emptiness allows, R_max = (3GM/(Lambda c^2))^(1/3).
"""
import sys, csv, math
import numpy as np
from scipy.optimize import curve_fit
G_SI, c = 6.674e-11, 2.998e8
MSUN = 1.989e30; AU_KM = 1.496e8
Lam = 1.1e-52

print("B1. Planets' rings and their outermost moons")
planets = [  # name, a (AU), mass (kg), outermost moon, a_moon (km), retrograde
    ("Earth", 1.000, 5.972e24, "Moon", 384_400, False),
    ("Mars", 1.524, 6.417e23, "Deimos", 23_457, False),
    ("Jupiter", 5.204, 1.898e27, "Kore", 24_203_300, True),
    ("Saturn", 9.583, 5.683e26, "S/2004 S 52", 26_446_400, True),
    ("Uranus", 19.19, 8.681e25, "Ferdinand", 20_421_400, True),
    ("Neptune", 30.07, 1.024e26, "S/2021 N 1", 50_700_200, True)]
for n, a, m, moon, am, retro in planets:
    RH = a * AU_KM * (m / (3 * MSUN)) ** (1 / 3)
    lim = 0.7 if retro else 0.5
    print(f"  {n:8s} ring {RH/1e6:8.2f} M km   outermost {moon:12s} {am/1e6:7.2f} M km = {am/RH:.2f} of ring "
          f"({'retrograde' if retro else 'prograde'}; stable limit ~{lim})  {'inside' if am < RH else 'OUTSIDE'}")

def eggleton(q):  # Roche-lobe radius / separation for mass ratio q = m1/m2
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + math.log(1 + q ** (1 / 3)))

folder = sys.argv[1]
rd = lambda f: list(csv.DictReader(open(f"{folder}/data/{f}")))
num = lambda s: float(s) if s not in ("", "nan", None) else float("nan")

print("\nB2. Satellites inside their galaxy's ring")
D_MW_M31 = 780.0
for host, col, mh, mo, f in (("Milky Way", "distance_gc", 1.0e12, 1.5e12, "dwarf_mw.csv"),
                             ("Andromeda", "distance_m31", 1.5e12, 1.0e12, "dwarf_m31.csv")):
    ring = eggleton(mh / mo) * D_MW_M31
    d = np.array([num(r[col]) for r in rd(f)]); d = d[np.isfinite(d)]
    print(f"  {host}: ring (share of the Local Group) {ring:.0f} kpc; {len(d)} satellites; "
          f"farthest {d.max():.0f} kpc; beyond the ring: {np.sum(d > ring)} "
          f"({', '.join(f'{x:.0f}' for x in sorted(d[d > ring]))} kpc)")

print("\nB3. Local Group balance ring from galaxy velocities")
rows = rd("dwarf_all.csv")
r = np.array([num(x["distance_lg"]) for x in rows]) / 1000.0   # Mpc
v = np.array([num(x["velocity_lg"]) for x in rows])
k = np.isfinite(r) & np.isfinite(v) & (r > 0.35) & (r < 3.0)
r, v = r[k], v[k]
print(f"  {k.sum()} galaxies between 0.35 and 3 Mpc from the Local Group centre")
model = lambda rr, H, R0: H * rr * (1 - (R0 / rr) ** 2)
p, pc = curve_fit(model, r, v, p0=(70, 1.0))
res = v - model(r, *p); s = np.std(res)
pcs = np.sqrt(np.diag(pc)) * max(1, s / np.sqrt(np.mean(res**2)))
print(f"  fit v = H r (1 - (R0/r)^2): R0 = {p[1]:.2f} +/- {pcs[1]:.2f} Mpc, H = {p[0]:.0f} km/s/Mpc, scatter {s:.0f} km/s")
# bootstrap
rng = np.random.default_rng(1); bs = []
for _ in range(2000):
    i = rng.integers(0, len(r), len(r))
    try: bs.append(curve_fit(model, r[i], v[i], p0=p)[0][1])
    except Exception: pass
lo, hi = np.percentile(bs, [16, 84])
print(f"  bootstrap R0 = {np.median(bs):.2f} (+{hi-np.median(bs):.2f}/-{np.median(bs)-lo:.2f}) Mpc  [Karachentsev et al. 2009: 0.96 +/- 0.03]")
for M in (1.9e12, 3e12, 5e12):
    Rmax = (3 * G_SI * M * MSUN / (Lam * c * c)) ** (1 / 3) / 3.0857e22
    print(f"  emptiness limit for M = {M:.1e} Suns: R_max = {Rmax:.2f} Mpc  -> measured ring {'inside' if np.median(bs) < Rmax else 'OUTSIDE'}")
# zero crossing nonparametric
order = np.argsort(r); edges = np.arange(0.35, 3.01, 0.25)
print("  median velocity by distance (Mpc: km/s, n):", "; ".join(
    f"{(a+b)/2:.2f}: {np.median(v[(r>=a)&(r<b)]):+.0f} ({np.sum((r>=a)&(r<b))})" for a, b in zip(edges[:-1], edges[1:]) if np.sum((r>=a)&(r<b)) >= 3))

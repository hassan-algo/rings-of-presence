"""
Dark matter as bumping sand (self-interacting dark matter), fitted to Milky Way and Andromeda satellites.

Rules (10c):
  speed-dependent bumping strength   s(v) = s0 / (1 + (v/w)^2)^2        [cm^2/g]
  bumps per grain over time t        N = rho * s(v_rel) * v_rel * t
      rho   = dark matter density inside the stars' half-light radius (from measured star speeds)
      v_rel = (4/sqrt(pi)) * sigma   mean relative speed of grains
  state:  N < 1          untouched  (ordinary cold dark matter)
          1 <= N < Nc    stirred into a low-density core   -> galaxy looks DIFFUSE for its size
          N >= Nc        core collapse (sped up by tides)   -> galaxy looks DENSE for its size
  limit from colliding clusters: s(2000 km/s) <= 2 cm^2/g

Part 1 fits (s0, w) to which satellites look dense / diffuse / typical for their size.
Part 2 traces Milky Way satellites through a Milky Way WITH a dark halo plus a moving LMC
(with dynamical friction) and asks whether dense satellites have closer passes (tides speeding
up collapse), as this picture predicts.
"""
import os, sys, math, csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite_predictions as sp

G = 4.30091e-6                       # kpc (km/s)^2 / Msun
MSUN_PC3 = 6.770e-23                 # g/cm^3 per Msun/pc^3
T_AGE = 10e9 * 3.15576e7             # s
TU = 0.977792                        # Gyr per kpc/(km/s)


# ---------------------------------------------------------------- data
def satellites(folder):
    out = []
    for g in sp.load(folder):
        r = g["raw"]
        if not r["vlos_sigma"] or r["vlos_sigma_ul"]:
            continue
        s = float(r["vlos_sigma"])
        r12 = 4 / 3 * g["Re"] / 1000.0
        M = 3 * s * s * r12 / G
        rho = M / (4 / 3 * math.pi * (r12 * 1000) ** 3)
        vrel = 4 / math.sqrt(math.pi) * s
        N1 = rho * MSUN_PC3 * vrel * 1e5 * T_AGE
        out.append(dict(key=g["key"], name=g["name"], host=g["host"], MV=g["MV"], r12=r12, sigma=s,
                        rho=rho, vrel=vrel, N1=N1, raw=r))
    lr, lrho = np.log10([o["r12"] for o in out]), np.log10([o["rho"] for o in out])
    slope, icpt = np.polyfit(lr, lrho, 1)
    for o in out:
        o["resid"] = math.log10(o["rho"]) - (icpt + slope * math.log10(o["r12"]))
        o["cls"] = "dense" if o["resid"] > 0.2 else ("diffuse" if o["resid"] < -0.2 else "typical")
    return out, slope


def s_of_v(v, s0, w):
    return s0 / (1 + (v / w) ** 2) ** 2


def state(N, Nc):
    return "untouched" if N < 1 else ("cored" if N < Nc else "collapsing")


def score(sats, s0, w, Nc):
    ok = 0
    for o in sats:
        st = state(o["N1"] * s_of_v(o["vrel"], s0, w), Nc)
        if o["cls"] == "dense":
            ok += st == "collapsing"
        elif o["cls"] == "diffuse":
            ok += st == "cored"
        else:
            ok += st != "collapsing"
    return ok / len(sats)


# ---------------------------------------------------------------- dark-matter Milky Way + LMC
RS, RHOS = 20.6, None
M200, C = 1.0e12, 10.0
RHOS = M200 / (4 * math.pi * RS**3 * (math.log(1 + C) - C / (1 + C)))
M_DISK, A_D, B_D = 6.8e10, 3.0, 0.28
M_BUL, A_B = 0.5e10, 0.5
M_LMC, A_LMC = 1.5e11, 20.0


def mw_acc(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    q = r / RS
    menc = 4 * math.pi * RHOS * RS**3 * (math.log(1 + q) - q / (1 + q))
    f = -G * menc / r**3
    ax, ay, az = f * x, f * y, f * z
    fb = -G * M_BUL / (r * (r + A_B) ** 2)
    ax, ay, az = ax + fb * x, ay + fb * y, az + fb * z
    zb = math.sqrt(z * z + B_D * B_D)
    den = (x * x + y * y + (A_D + zb) ** 2) ** 1.5
    ax += -G * M_DISK * x / den
    ay += -G * M_DISK * y / den
    az += -G * M_DISK * z * (A_D + zb) / (zb * den)
    return ax, ay, az


def nfw_density(r):
    q = r / RS
    return RHOS / (q * (1 + q) ** 2)


def lmc_rhs(t, w):
    x, y, z, vx, vy, vz = w
    ax, ay, az = mw_acc(x, y, z)
    r = math.sqrt(x * x + y * y + z * z)
    v = math.sqrt(vx * vx + vy * vy + vz * vz)
    vc = math.sqrt(G * 4 * math.pi * RHOS * RS**3 * (math.log(1 + r / RS) - (r / RS) / (1 + r / RS)) / r)
    sig = vc / math.sqrt(2)
    X = v / (math.sqrt(2) * sig)
    df = -4 * math.pi * G**2 * M_LMC * nfw_density(r) * 3.0 * (math.erf(X) - 2 * X / math.sqrt(math.pi) * math.exp(-X * X)) / v**3
    return [vx, vy, vz, ax + df * vx, ay + df * vy, az + df * vz]


def galcen(r):
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Galactocentric
    c = SkyCoord(ra=float(r["ra"]) * u.deg, dec=float(r["dec"]) * u.deg, distance=float(r["distance"]) * u.kpc,
                 pm_ra_cosdec=float(r["pmra"]) * u.mas / u.yr, pm_dec=float(r["pmdec"]) * u.mas / u.yr,
                 radial_velocity=float(r["vlos_systemic"]) * u.km / u.s, frame="icrs")
    g = c.transform_to(Galactocentric())
    return [g.x.to(u.kpc).value, g.y.to(u.kpc).value, g.z.to(u.kpc).value,
            g.v_x.to(u.km / u.s).value, g.v_y.to(u.km / u.s).value, g.v_z.to(u.km / u.s).value]


def pericentres(folder, sats):
    rows = {r["key"]: r for r in csv.DictReader(open(f"{folder}/data/dwarf_mw.csv"))}
    T = 4.0 / TU
    lmc = solve_ivp(lmc_rhs, (0, -T), galcen(rows["lmc"]), method="DOP853", rtol=1e-9, atol=1e-9, dense_output=True)
    ts = np.linspace(0, -T, 4001)
    dl = np.linalg.norm(lmc.sol(ts)[:3].T, axis=1)
    i = next((k for k in range(1, len(dl) - 1) if dl[k] < dl[k - 1] and dl[k] <= dl[k + 1]), int(np.argmin(dl)))
    print(f"LMC (with friction): now {dl[0]:.0f} kpc; most recent closest pass {dl[i]:.0f} kpc, "
          f"{-ts[i] * TU * 1000:.0f} Myr ago; 4 Gyr ago at {dl[-1]:.0f} kpc")

    def sat_rhs(t, w):
        x, y, z = w[:3]
        ax, ay, az = mw_acc(x, y, z)
        lx, ly, lz = lmc.sol(t)[:3]
        dx, dy, dz = x - lx, y - ly, z - lz
        rl = math.sqrt(dx * dx + dy * dy + dz * dz)
        f = -G * M_LMC / (rl * (rl + A_LMC) ** 2)
        return [w[3], w[4], w[5], ax + f * dx, ay + f * dy, az + f * dz]

    for o in sats:
        r = o["raw"]
        if o["host"] != "mw" or not (r["pmra"] and r["pmdec"] and r["vlos_systemic"]):
            continue
        sol = solve_ivp(sat_rhs, (0, -T), galcen(r), method="DOP853", rtol=1e-9, atol=1e-9, dense_output=True)
        d = np.linalg.norm(sol.sol(ts)[:3].T, axis=1)
        mins = [k for k in range(1, len(d) - 1) if d[k] < d[k - 1] and d[k] <= d[k + 1]]
        k = mins[0] if mins else int(np.argmin(d))
        o["peri"], o["peri_t"], o["dnow"] = d[k], -ts[k] * TU, d[0]


if __name__ == "__main__":
    folder = sys.argv[1]
    sats, slope = satellites(folder)
    counts = {c: sum(o["cls"] == c for o in sats) for c in ("dense", "typical", "diffuse")}
    print(f"{len(sats)} satellites; density vs size slope {slope:.2f}; classes: {counts}")

    # ---------------- Part 1: scan the bumping law
    s0s = np.geomspace(0.1, 5000, 90)
    ws = np.geomspace(1, 2000, 90)
    print("\nPart 1: best bumping laws (score = fraction of satellites whose state matches their look)")
    best_all = []
    for Nc in (100, 300, 1000):
        best = (-1, None, None)
        grid = np.zeros((len(s0s), len(ws)))
        for i, s0 in enumerate(s0s):
            for j, w in enumerate(ws):
                if s_of_v(2000, s0, w) > 2:          # cluster limit
                    grid[i, j] = np.nan
                    continue
                grid[i, j] = score(sats, s0, w, Nc)
                if grid[i, j] > best[0]:
                    best = (grid[i, j], s0, w)
        base = score(sats, 1e-9, 1.0, Nc)            # no bumping at all (cold dark matter)
        good = np.argwhere(grid >= best[0] - 0.03)
        print(f"  collapse threshold Nc={Nc:4d}: best score {best[0] * 100:.0f}% at s0 = {best[1]:.0f} cm^2/g, "
              f"w = {best[2]:.1f} km/s   (no bumping: {base * 100:.0f}%)")
        print(f"     within 3 points of best: s0 {s0s[good[:, 0]].min():.0f}-{s0s[good[:, 0]].max():.0f} cm^2/g, "
              f"w {ws[good[:, 1]].min():.1f}-{ws[good[:, 1]].max():.1f} km/s")
        best_all.append((Nc, best))

    Nc, (sc, s0, w) = best_all[1]
    print(f"\nUsing Nc = {Nc}, s0 = {s0:.0f}, w = {w:.1f} km/s:")
    for v in (5, 10, 20, 50, 200, 2000):
        print(f"  bumping strength at {v:5d} km/s: {s_of_v(v, s0, w):10.4f} cm^2/g")
    print(f"\n  {'satellite':18s} {'size pc':>7s} {'speed':>6s} {'looks':>8s} {'bumps N':>9s} {'state':>11s} {'match':>6s}")
    for o in sorted(sats, key=lambda o: o["resid"], reverse=True):
        N = o["N1"] * s_of_v(o["vrel"], s0, w)
        st = state(N, Nc)
        m = (st == "collapsing") if o["cls"] == "dense" else ((st == "cored") if o["cls"] == "diffuse" else st != "collapsing")
        o["N"], o["state"], o["match"] = N, st, m
        if o["name"] in ("Draco", "Crater II", "Segue 1", "Willman 1", "Fornax", "Sculptor", "Antlia II", "Sextans",
                         "Ursa Minor", "Andromeda XIX", "Tucana II", "Leo I", "Carina", "Bootes I", "Reticulum II"):
            print(f"  {o['name'][:18]:18s} {o['r12'] * 1000:7.0f} {o['sigma']:6.2f} {o['cls']:>8s} {N:9.1f} {st:>11s} "
                  f"{'yes' if m else 'no':>6s}")
    for c in ("dense", "typical", "diffuse"):
        sel = [o for o in sats if o["cls"] == c]
        print(f"  {c:8s}: {sum(o['match'] for o in sel)}/{len(sel)} match")

    # ---------------- Part 2: orbits in a dark-matter Milky Way
    print("\nPart 2: closest passes in a Milky Way with a dark halo (1e12 Msun) + LMC (1.5e11 Msun)")
    pericentres(folder, sats)
    mw = [o for o in sats if "peri" in o]
    rho_res = np.array([o["resid"] for o in mw])
    peri = np.array([o["peri"] for o in mw])
    rs, p = spearmanr(rho_res, peri)
    print(f"  {len(mw)} satellites: density-for-size vs closest pass, rank correlation {rs:+.2f} (p = {p:.3f})")
    bright = [o for o in mw if o["MV"] < -7.7]
    rs2, p2 = spearmanr([o["resid"] for o in bright], [o["peri"] for o in bright])
    print(f"  bright ones only ({len(bright)}): {rs2:+.2f} (p = {p2:.3f})")
    print("  bumping sand + tides predicts NEGATIVE (dense <-> close passes); "
          "ordinary dark matter + tides predicts POSITIVE (close passes strip and thin them)")
    for o in sorted(bright, key=lambda o: o["peri"]):
        print(f"    {o['name']:18s} closest {o['peri']:5.0f} kpc ({o['peri_t']:.1f} Gyr ago)   looks {o['cls']:8s} "
              f"(density-for-size {o['resid']:+.2f})")

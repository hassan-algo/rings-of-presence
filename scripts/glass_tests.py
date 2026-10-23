"""
"Hot things turn sand into glass": do galaxies whose dark matter centres were heated have smoother
(lower-density) centres?

Test A (heat from star formation, rule 10e):
    rho_DM(150 pc) vs stars per dark matter (M*/M200) and vs how long star formation lasted.
    Data: Read, Walker & Steger 2019, Table 1 (8 dwarf spheroidals + 8 dwarf irregulars).
    Prediction: more stars per dark matter / longer star formation -> lower central density.

Test B (fast movers):
    For Milky Way satellites, current speed and speed at closest pass in a dark-matter Milky Way
    + moving LMC (same orbit code as sand_fit.py). Do fast movers have smoother centres?
    B1: the 8 dwarf spheroidals with measured rho_DM(150 pc)
    B2: all satellites with 3-D motions, using density-for-size from star speeds
"""
import os, sys, math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sand_fit as sf

# name, M* (1e6 Msun), M200 (1e9 Msun), rho150 (1e8 Msun/kpc^3), -err, +err, t_trunc (Gyr ago), type
READ19 = [
    ("Ursa Minor", 0.29, 2.8, 1.53, 0.32, 0.35, 12.4, "dSph"),
    ("Draco", 0.29, 1.8, 2.36, 0.29, 0.29, 11.7, "dSph"),
    ("Sculptor", 2.3, 5.7, 1.49, 0.23, 0.28, 11.8, "dSph"),
    ("Sextans", 0.44, 2.0, 1.28, 0.29, 0.34, 10.6, "dSph"),
    ("Leo I", 5.5, 5.6, 1.77, 0.34, 0.33, 3.1, "dSph"),
    ("Leo II", 0.74, 1.6, 1.84, 0.16, 0.17, 6.3, "dSph"),
    ("Carina", 0.38, 0.8, 1.16, 0.22, 0.20, 2.8, "dSph"),
    ("Fornax", 43, 21.9, 0.79, 0.19, 0.27, 1.75, "dSph"),
    ("WLM", 16.2, 8.3, 0.52, 0.09, 0.09, 0, "dIrr"),
    ("DDO 52", 52.7, 12, 0.38, 0.10, 0.17, 0, "dIrr"),
    ("DDO 87", 33, 11.3, 0.31, 0.09, 0.18, 0, "dIrr"),
    ("DDO 154", 8.35, 12.6, 0.46, 0.10, 0.13, 0, "dIrr"),
    ("Aquarius", 0.68, 0.68, 0.36, 0.19, 0.22, 0, "dIrr"),
    ("NGC 2366", 69.5, 24, 0.18, 0.03, 0.05, 0, "dIrr"),
    ("CVnIdwA", 4.1, 1.7, 0.33, 0.09, 0.12, 0, "dIrr"),
    ("DDO 168", 59, 21, 0.31, 0.07, 0.11, 0, "dIrr"),
]
TU = sf.TU


def orbits(folder, sats):
    """Current speed, closest-pass distance and speed at closest pass, Milky Way with dark halo + LMC."""
    import csv
    rows = {r["key"]: r for r in csv.DictReader(open(f"{folder}/data/dwarf_mw.csv"))}
    T = 4.0 / TU
    lmc = solve_ivp(sf.lmc_rhs, (0, -T), sf.galcen(rows["lmc"]), method="DOP853",
                    rtol=1e-9, atol=1e-9, dense_output=True)
    ts = np.linspace(0, -T, 4001)

    def rhs(t, w):
        x, y, z = w[:3]
        ax, ay, az = sf.mw_acc(x, y, z)
        lx, ly, lz = lmc.sol(t)[:3]
        dx, dy, dz = x - lx, y - ly, z - lz
        rl = math.sqrt(dx * dx + dy * dy + dz * dz)
        f = -sf.G * sf.M_LMC / (rl * (rl + sf.A_LMC) ** 2)
        return [w[3], w[4], w[5], ax + f * dx, ay + f * dy, az + f * dz]

    for o in sats:
        r = o["raw"]
        if o["host"] != "mw" or not (r["pmra"] and r["pmdec"] and r["vlos_systemic"]):
            continue
        w0 = sf.galcen(r)
        sol = solve_ivp(rhs, (0, -T), w0, method="DOP853", rtol=1e-9, atol=1e-9, dense_output=True)
        W = sol.sol(ts)
        d = np.linalg.norm(W[:3].T, axis=1)
        v = np.linalg.norm(W[3:].T, axis=1)
        mins = [k for k in range(1, len(d) - 1) if d[k] < d[k - 1] and d[k] <= d[k + 1]]
        k = mins[0] if mins else int(np.argmin(d))
        o.update(v_now=v[0], peri=d[k], v_peri=v[k], v_max=v.max())


def corr(label, x, y):
    rs, p = spearmanr(x, y)
    print(f"  {label:62s} rank correlation {rs:+.2f}  (p = {p:.3f}, n = {len(x)})")
    return rs, p


if __name__ == "__main__":
    folder = sys.argv[1]
    print("=== Test A: heat from star formation (Read, Walker & Steger 2019 data) ===")
    ratio = np.array([g[1] * 1e6 / (g[2] * 1e9) for g in READ19])
    rho = np.array([g[3] for g in READ19])
    ttr = np.array([g[6] for g in READ19])
    for g, q in sorted(zip(READ19, ratio), key=lambda z: z[1]):
        print(f"  {g[0]:11s} {g[7]:4s} stars/dark {q * 100:6.3f}%   centre density {g[3]:.2f} x1e8 Msun/kpc^3   "
              f"star formation stopped {g[6]:4.1f} Gyr ago")
    corr("stars per dark matter vs centre density (all 16)", ratio, rho)
    corr("how long ago star formation stopped vs centre density (all 16)", ttr, rho)
    sph = [i for i, g in enumerate(READ19) if g[7] == "dSph"]
    corr("stars per dark matter vs centre density (8 satellites only)", ratio[sph], rho[sph])
    early = rho[ttr > 6]; ext = rho[ttr < 3]
    print(f"  stopped >6 Gyr ago: mean centre density {early.mean():.2f};  still/recently forming (<3 Gyr): "
          f"{ext.mean():.2f}  (x1e8 Msun/kpc^3; ordinary dense centre ~ above 1)")

    print("\n=== Test B: fast movers (Milky Way with dark halo + LMC) ===")
    sats, _ = sf.satellites(folder)
    orbits(folder, sats)
    withorb = [o for o in sats if "v_now" in o]
    byname = {o["name"]: o for o in withorb}
    b1 = [(g, byname[g[0]]) for g in READ19 if g[7] == "dSph" and g[0] in byname]
    print("  B1: the 8 satellites with measured centre density")
    for g, o in sorted(b1, key=lambda z: -z[1]["v_peri"]):
        print(f"    {g[0]:11s} speed now {o['v_now']:4.0f} km/s   at closest pass {o['v_peri']:4.0f} km/s "
              f"({o['peri']:4.0f} kpc)   centre density {g[3]:.2f}")
    corr("speed now vs centre density", [o["v_now"] for g, o in b1], [g[3] for g, o in b1])
    corr("speed at closest pass vs centre density", [o["v_peri"] for g, o in b1], [g[3] for g, o in b1])
    print(f"  B2: all {len(withorb)} satellites with 3-D motions (density-for-size from star speeds)")
    corr("speed now vs density-for-size", [o["v_now"] for o in withorb], [o["resid"] for o in withorb])
    corr("speed at closest pass vs density-for-size", [o["v_peri"] for o in withorb], [o["resid"] for o in withorb])
    fast = [o for o in withorb if o["v_peri"] > np.median([q["v_peri"] for q in withorb])]
    slow = [o for o in withorb if o not in fast]
    print(f"  fastest half: median density-for-size {np.median([o['resid'] for o in fast]):+.2f};  "
          f"slowest half: {np.median([o['resid'] for o in slow]):+.2f}  (negative = smoother/more spread out)")

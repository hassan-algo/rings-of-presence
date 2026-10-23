"""
Rule 13 with time: each satellite's star speeds are set by the outside presence it felt
over its recent history, not just today.

1. Measured sky position, distance, line-of-sight speed and Gaia proper motion -> 3-D position
   and velocity around the Milky Way (astropy Galactocentric frame).
2. Trace the Large Magellanic Cloud and every satellite backward 3 Gyr through the rule-13 field
   of the Milky Way + the moving LMC:
       g_N = sum of Hernquist pulls (MW 6e10 Msun, a = 3 kpc; LMC 3.2e9 Msun, a = 1.5 kpc)
       g   = g_N * nu(|g_N| / a0),   nu(y) = 1/(1 - exp(-sqrt y)),   a0 = 1.13e-10 m/s^2
3. Outside pull averaged over each satellite's crossing time (iterated, using predicted speeds):
       g_out,eff = (1/t_cross) * integral of |g(x_sat(t))| dt over the last t_cross
4. Predict star speeds (same formulas as before) and flag tidally stretched satellites:
       r_tidal = D_peri * (M_sat,eff / (3 M_host,eff))^(1/3),  M_host,eff = g(D_peri) D_peri^2 / G,
       M_sat,eff = 3 sigma^2 r_half / G;   stretched if r_tidal < 2 r_half
Units inside: kpc, km/s, Gyr-like (kpc/(km/s) = 0.9778 Gyr), Msun.
"""
import os, sys, csv, math
import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satellite_predictions as sp

G = 4.30091e-6                      # kpc (km/s)^2 / Msun
ACC_SI = 1e6 / 3.0856776e19         # (km/s)^2/kpc -> m/s^2
A0 = 1.13e-10 / ACC_SI              # in (km/s)^2/kpc
TU = 0.977792                       # Gyr per time unit
M_MW, A_MW = 6.0e10, 3.0
M_LMC, A_LMC = 3.2e9, 1.5
T_BACK = 3.0 / TU


def nu(y):
    return 1.0 / (-math.expm1(-math.sqrt(max(y, 1e-30))))


def hern(dx, M, a):
    r = math.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)
    f = -G * M / (r * (r + a) ** 2)
    return (f * dx[0], f * dx[1], f * dx[2])


def galcen(r):
    c = SkyCoord(ra=float(r["ra"]) * u.deg, dec=float(r["dec"]) * u.deg,
                 distance=float(r["distance"]) * u.kpc,
                 pm_ra_cosdec=float(r["pmra"]) * u.mas / u.yr, pm_dec=float(r["pmdec"]) * u.mas / u.yr,
                 radial_velocity=float(r["vlos_systemic"]) * u.km / u.s, frame="icrs")
    g = c.transform_to(Galactocentric())
    return np.array([g.x.to(u.kpc).value, g.y.to(u.kpc).value, g.z.to(u.kpc).value,
                     g.v_x.to(u.km / u.s).value, g.v_y.to(u.km / u.s).value, g.v_z.to(u.km / u.s).value])


def field(x, lmc_pos):
    gx, gy, gz = hern(x, M_MW, A_MW)
    if lmc_pos is not None:
        lx, ly, lz = hern((x[0] - lmc_pos[0], x[1] - lmc_pos[1], x[2] - lmc_pos[2]), M_LMC, A_LMC)
        gx, gy, gz = gx + lx, gy + ly, gz + lz
    gn = math.sqrt(gx * gx + gy * gy + gz * gz)
    k = nu(gn / A0)
    return (gx * k, gy * k, gz * k), gn * k


def integrate(w0, lmc_sol=None):
    def rhs(t, w):
        lp = lmc_sol.sol(t)[:3] if lmc_sol is not None else None
        (ax, ay, az), _ = field(w[:3], lp)
        return [w[3], w[4], w[5], ax, ay, az]
    return solve_ivp(rhs, (0, -T_BACK), w0, method="DOP853", rtol=1e-9, atol=1e-9, dense_output=True)


def predict_sigma(M_star, r12, g_out):
    """Same rule-13 formulas as Test 3, with g_out in (km/s)^2/kpc; returns km/s."""
    s_iso = (4 / 81 * G * M_star * A0) ** 0.25
    g_in = 3 * s_iso**2 / r12
    if g_in >= g_out:
        return s_iso
    return min(math.sqrt(A0 * G * M_star / (3 * g_out * r12)), s_iso)


if __name__ == "__main__":
    folder = sys.argv[1]
    rows = {r["key"]: r for r in csv.DictReader(open(f"{folder}/data/dwarf_mw.csv"))}
    lmc = galcen(rows["lmc"])
    lmc_sol = integrate(lmc)
    ts = np.linspace(0, -T_BACK, 3001)
    lmc_d = np.linalg.norm(lmc_sol.sol(ts)[:3].T, axis=1)
    print(f"LMC now {lmc_d[0]:.1f} kpc from the Galactic centre, speed {np.linalg.norm(lmc[3:]):.0f} km/s; "
          f"closest in last 3 Gyr {lmc_d.min():.1f} kpc, farthest {lmc_d.max():.0f} kpc")

    out = []
    for g in sp.load(folder):
        if g["host"] != "mw":
            continue
        r = g["raw"]
        if not (r["pmra"] and r["pmdec"] and r["vlos_systemic"] and r["vlos_sigma"]) or r["vlos_sigma_ul"]:
            continue
        w0 = galcen(r)
        sol = integrate(w0, lmc_sol)
        W = sol.sol(ts)
        pos = W[:3].T
        lp = lmc_sol.sol(ts)[:3].T
        gout_t = np.array([field(p, l)[1] for p, l in zip(pos, lp)])
        D = np.linalg.norm(pos, axis=1)
        d_lmc_now = np.linalg.norm(pos[0] - lp[0])

        L = 10 ** (-0.4 * (g["MV"] - 4.83))
        Mst = 2 * L
        r12 = 4 / 3 * g["Re"] / 1000.0                     # kpc
        g_snap = gout_t[0]
        s_snap = predict_sigma(Mst, r12, g_snap)
        s = s_snap
        for _ in range(5):                                  # crossing time from predicted speed
            t_cross = r12 / s                               # time units
            m = ts >= -t_cross
            g_eff = np.trapezoid(gout_t[m], -ts[m]) / max(t_cross, 1e-9) if m.sum() > 1 else g_snap
            s = predict_sigma(Mst, r12, g_eff)

        # most recent closest approach to the Milky Way
        mins = [i for i in range(1, len(D) - 1) if D[i] < D[i - 1] and D[i] <= D[i + 1]]
        i_p = mins[0] if mins else int(np.argmin(D))
        D_p, t_p = D[i_p], -ts[i_p] * TU
        g_host_p = field(pos[i_p], None)[1]
        M_host_eff = g_host_p * D_p**2 / G
        M_sat_eff = 3 * s**2 * r12 / G
        r_t = D_p * (M_sat_eff / (3 * M_host_eff)) ** (1 / 3)
        stretched = r_t < 2 * r12

        # prediction uncertainty: star weight 1-4 (vs 2) counted as +/- 1 sigma, in each regime
        lo, hi = predict_sigma(0.5 * Mst, r12, g_eff), predict_sigma(2 * Mst, r12, g_eff)
        lo_s, hi_s = predict_sigma(0.5 * Mst, r12, g_snap), predict_sigma(2 * Mst, r12, g_snap)
        perr_h = 0.5 * (math.log10(hi) - math.log10(lo))
        perr_s = 0.5 * (math.log10(hi_s) - math.log10(lo_s))
        s_obs = float(r["vlos_sigma"])
        err = max((float(r["vlos_sigma_em"] or 0) + float(r["vlos_sigma_ep"] or 0)) / 2, 0.05 * s_obs)
        out.append(dict(name=g["name"], MV=g["MV"], Dnow=D[0], Dp=D_p, tp=t_p, dlmc=d_lmc_now,
                        tc=r12 / s * TU * 1000, gs=g_snap * ACC_SI / 1.13e-10, ge=g_eff * ACC_SI / 1.13e-10,
                        s_snap=s_snap, s_hist=s, obs=s_obs, err=err, rt=r_t, r12=r12, stretched=stretched,
                        perr_snap=perr_s, perr_hist=perr_h))

    out.sort(key=lambda o: o["MV"])
    print(f"\n{len(out)} Milky Way satellites with full 3-D motions\n")
    print(f"{'satellite':18s} {'D now':>6s} {'closest':>8s} {'when':>6s} {'LMC':>6s} {'cross':>6s} "
          f"{'pull now':>8s} {'pull avg':>8s} {'snapshot':>8s} {'history':>8s} {'measured':>12s} {'tides':>9s}")
    for o in out:
        print(f"{o['name'][:18]:18s} {o['Dnow']:6.0f} {o['Dp']:6.0f}kpc {o['tp']:5.2f}G {o['dlmc']:6.0f} "
              f"{o['tc']:5.0f}M {o['gs']:8.3f} {o['ge']:8.3f} {o['s_snap']:8.2f} {o['s_hist']:8.2f} "
              f"{o['obs']:6.2f}±{o['err']:4.2f} {'STRETCHED' if o['stretched'] else 'ok':>9s}")

    def summary(label, sel, key):
        if not sel:
            return
        pk = "perr_snap" if key == "s_snap" else "perr_hist"
        d = np.array([(math.log10(o["obs"]) - math.log10(o[key])) /
                      math.hypot(o[pk], o["err"] / (o["obs"] * math.log(10))) for o in sel])
        ratio = np.array([o["obs"] / o[key] for o in sel])
        print(f"  {label:48s} n={len(sel):2d}  within 2 sigma {np.mean(np.abs(d) < 2) * 100:3.0f}%   "
              f"median measured/predicted {np.median(ratio):.2f}")

    ch = np.array([o["s_hist"] / o["s_snap"] - 1 for o in out]) * 100
    print(f"\nChange from adding history + LMC: median {np.median(ch):+.1f}%, range {ch.min():+.0f}% to {ch.max():+.0f}%")
    print(f"Tidally stretched (r_tidal < 2 r_half at last closest approach): "
          f"{sum(o['stretched'] for o in out)} of {len(out)}")
    print("\nSummary (prediction uncertainty: star weight 1-4 as +/- 1 sigma):")
    for key, lab in (("s_snap", "snapshot"), ("s_hist", "with history + LMC")):
        summary(f"all, {lab}", out, key)
        summary(f"not tidally stretched, {lab}", [o for o in out if not o["stretched"]], key)
        summary(f"bright (M_V < -7.7), not stretched, {lab}",
                [o for o in out if not o["stretched"] and o["MV"] < -7.7], key)

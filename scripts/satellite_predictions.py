"""
Test 3: predict star speeds (velocity dispersion) in satellite galaxies of the Milky Way and Andromeda
from their LIGHT, SIZE and DISTANCE only (rule 13), then compare with the measured speeds.

Inputs come from the Local Volume Database (Pace 2024, github.com/apace7/local_volume_database).
Step 1 computes predictions without reading any measured speed; step 2 loads the measurements.

Rule 13 formulas (a0 = 1.13e-10 m/s^2, nu(y) = 1/(1 - exp(-sqrt y))):
  M      = Y * L                         star mass, Y = 2 (range 1-4) solar units per solar luminosity
  r_half = 4/3 * R_e                     3-D half-light radius
  host pull at the satellite: g_out = g_N * nu(g_N / a0),  g_N = G M_host / D^2
  isolated:          sigma^4 = (4/81) G M a0
  with outside pull: sigma^2 = a0 G M / (3 g_out r_half)      used when g_out > g_in
  g_in = 3 sigma_iso^2 / r_half
"""
import csv, math, sys
import numpy as np

G = 6.67430e-11
M_SUN = 1.98847e30
PC = 3.0856776e16
KPC = 1000 * PC
A0 = 1.13e-10
HOSTS = {"mw": 6.0e10, "m_031": 1.2e11}     # baryonic masses (solar masses), assumed


def nu(y):
    return 1.0 / (-math.expm1(-math.sqrt(y)))


def predict(MV, Re_pc, D_kpc, host, ups):
    L = 10 ** (-0.4 * (MV - 4.83))
    M = ups * L * M_SUN
    r12 = 4 / 3 * Re_pc * PC
    gN_host = G * HOSTS[host] * M_SUN / (D_kpc * KPC) ** 2
    g_out = gN_host * nu(gN_host / A0)
    s_iso = (4 / 81 * G * M * A0) ** 0.25
    g_in = 3 * s_iso**2 / r12
    s_efe = math.sqrt(A0 * G * M / (3 * g_out * r12))
    s13 = s_iso if g_in >= g_out else min(s_efe, s_iso)
    return s13 / 1e3, s_iso / 1e3, g_in / A0, g_out / A0


def load(folder):
    rows = []
    for f, host in (("dwarf_mw.csv", "mw"), ("dwarf_m31.csv", "m_031")):
        for r in csv.DictReader(open(f"{folder}/data/{f}")):
            try:
                MV, Re, D = float(r["M_V"]), float(r["rhalf_sph_physical"]), float(r["distance_host"])
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in (MV, Re, D)):
                continue
            if r["confirmed_galaxy"] != "1" or r["key"] in ("lmc", "smc", "sagittarius_1", "m_032", "ngc_0205"):
                continue                               # rotating or being torn apart, or not dispersion-supported
            rows.append(dict(key=r["key"], name=r["name"], host=host, MV=MV, Re=Re, D=D, raw=r))
    return rows


if __name__ == "__main__":
    folder = sys.argv[1]
    gals = load(folder)

    # ---------------- step 1: predictions only
    preds = []
    for g in gals:
        s2, iso2, gin, gout = predict(g["MV"], g["Re"], g["D"], g["host"], 2)
        s1, _, _, _ = predict(g["MV"], g["Re"], g["D"], g["host"], 1)
        s4, _, _, _ = predict(g["MV"], g["Re"], g["D"], g["host"], 4)
        preds.append(dict(g, s=s2, lo=s1, hi=s4, iso=iso2, gin=gin, gout=gout,
                          regime="outside pull" if gout > gin else "isolated"))
    with open("satellite_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "host", "M_V", "R_e_pc", "D_kpc", "g_in_a0", "g_out_a0", "regime",
                    "sigma_rule13", "sigma_low_Y1", "sigma_high_Y4", "sigma_if_isolated"])
        for p in preds:
            w.writerow([p["name"], p["host"], p["MV"], p["Re"], p["D"], f"{p['gin']:.4f}", f"{p['gout']:.4f}",
                        p["regime"], f"{p['s']:.2f}", f"{p['lo']:.2f}", f"{p['hi']:.2f}", f"{p['iso']:.2f}"])
    print(f"Step 1: wrote predictions for {len(preds)} satellites (no measured speeds used)\n")

    # ---------------- step 2: compare with measurements
    out = []
    for p in preds:
        r = p["raw"]
        if not r["vlos_sigma"] or r["vlos_sigma_ul"]:
            continue
        s, em, ep = float(r["vlos_sigma"]), float(r["vlos_sigma_em"] or 0), float(r["vlos_sigma_ep"] or 0)
        err = max((em + ep) / 2, 0.05 * s)
        # uncertainty of the prediction from the star-weight range, in log space
        pred_err_log = 0.5 * (math.log10(p["hi"]) - math.log10(p["lo"])) / 2
        obs_err_log = err / (s * math.log(10))
        tot = math.hypot(pred_err_log, obs_err_log)
        d13 = (math.log10(s) - math.log10(p["s"])) / tot
        diso = (math.log10(s) - math.log10(p["iso"])) / tot
        out.append(dict(p, obs=s, err=err, d13=d13, diso=diso,
                        faint=p["MV"] > -7.7))
    out.sort(key=lambda o: (o["host"], o["MV"]))
    print(f"{'satellite':22s} {'host':5s} {'M_V':>6s} {'D kpc':>6s} {'regime':13s} "
          f"{'rule13':>7s} {'range':>11s} {'isolated':>8s} {'measured':>13s} {'off(σ)':>7s}")
    for o in out:
        print(f"{o['name'][:22]:22s} {o['host'][:5]:5s} {o['MV']:6.2f} {o['D']:6.0f} {o['regime']:13s} "
              f"{o['s']:7.2f} {o['lo']:5.2f}-{o['hi']:5.2f} {o['iso']:8.2f} {o['obs']:7.2f}±{o['err']:4.2f} {o['d13']:+7.1f}")

    for label, sel in (("Classical (M_V < -7.7)", [o for o in out if not o["faint"]]),
                       ("Ultra-faint (M_V > -7.7)", [o for o in out if o["faint"]]),
                       ("Outside-pull regime only", [o for o in out if o["regime"] == "outside pull"])):
        if not sel:
            continue
        d13 = np.array([o["d13"] for o in sel]); diso = np.array([o["diso"] for o in sel])
        lr13 = np.array([math.log10(o["obs"] / o["s"]) for o in sel])
        lriso = np.array([math.log10(o["obs"] / o["iso"]) for o in sel])
        print(f"\n{label}: {len(sel)} satellites")
        print(f"  rule 13:          within 2 sigma {np.mean(np.abs(d13) < 2) * 100:3.0f}%   "
              f"median measured/predicted {10 ** np.median(lr13):.2f}   chi2/N {np.mean(d13 ** 2):.1f}")
        print(f"  no outside pull:  within 2 sigma {np.mean(np.abs(diso) < 2) * 100:3.0f}%   "
              f"median measured/predicted {10 ** np.median(lriso):.2f}   chi2/N {np.mean(diso ** 2):.1f}")

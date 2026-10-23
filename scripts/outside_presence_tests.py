"""
Rule 13 -- others' presence changes what we look at (external field effect) -- plus
the photon-watched-vs-unwatched test.

Rule 13 (one-dimensional form):
    g_inside = (g_N + g_out) * nu((g_N + g_out)/a0)  -  g_out * nu(g_out/a0)
    nu(y) = 1 / (1 - exp(-sqrt(y)))          (the boost function fitted to galaxies)
    a0 = 1.13e-10 m/s^2                      (our SPARC fit)
In Einstein + dark matter, g_out has no effect on internal motion (apart from tides).

Usage: python3 outside_presence_tests.py <folder with SPARC *_rotmod.dat files>
"""
import os, sys, math
import numpy as np

G = 6.67430e-11
c = 299_792_458.0
HBAR = 1.054571817e-34
M_SUN = 1.98847e30
AU = 1.495978707e11
KPC = 3.0856776e19
A0 = 1.13e-10


def nu(y):
    y = np.maximum(y, 1e-30)
    return 1.0 / (-np.expm1(-np.sqrt(y)))


def dnu_dy(y):
    s = np.sqrt(y)
    return -nu(y) ** 2 * np.exp(-s) / (2 * s)


def g_rule13(gN, gout, a0=A0):
    tot = gN + gout
    return tot * nu(tot / a0) - gout * nu(gout / a0)


# ---------------------------------------------------------------- A. Wide binaries near the Sun
def wide_binaries():
    print("=== A. Wide star pairs near the Sun ===")
    v_sun, R_sun = 233e3, 8.2 * KPC
    g_out = v_sun**2 / R_sun
    ye = g_out / A0
    nu_e = float(nu(ye))
    L_e = float(ye * dnu_dy(ye) / nu_e)            # d ln nu / d ln y
    along, across = nu_e * (1 + L_e), nu_e
    avg = nu_e * (1 + L_e / 3)
    print(f"Galaxy's pull at the Sun: {g_out:.2e} m/s^2 = {ye:.2f} a0")
    print(f"Far-apart pairs: pull boosted x{along:.3f} along the Galaxy's pull, x{across:.3f} across it, "
          f"~x{avg:.3f} on average")
    print(f"  -> orbital speed boost: {100 * (math.sqrt(along) - 1):.1f}% / {100 * (math.sqrt(across) - 1):.1f}% / "
          f"~{100 * (math.sqrt(avg) - 1):.1f}%")
    # same boost the galaxy fit requires at this pull (no outside field in the fit itself)
    print(f"Extra pull our galaxy fit gives at g = {ye:.2f} a0 (galaxies): x{nu_e:.3f}")
    M = 1.5 * M_SUN
    rows = []

    def boosted(g):                          # rule-13 field for a signed 1-D total pull
        return g * nu(abs(g) / A0)

    print("  (relative pull between the two stars, approximate algebraic form)")
    for s_kau in (1, 2, 3, 5, 7, 10, 15, 20, 30):
        s = s_kau * 1000 * AU
        gN = G * M / s**2                    # Newtonian relative pull
        h = gN / 2                           # each star's pull toward the other (equal masses)
        along = (boosted(g_out + h) - boosted(g_out - h)) / (2 * h)
        across = float(nu(math.hypot(g_out, h) / A0))
        avg = (along + 2 * across) / 3
        iso = float(nu(h / A0))
        rows.append((s_kau, gN / A0, along, across, avg, iso))
        sp = lambda b: 100 * (math.sqrt(b) - 1)
        print(f"  separation {s_kau:>3} kAU: inside pull {gN / A0:6.2f} a0   speed boost: along {sp(along):5.1f}%  "
              f"across {sp(across):5.1f}%  average {sp(avg):5.1f}%   (if no outside pull: {sp(iso):5.1f}%)")
    return rows


# ---------------------------------------------------------------- B. SPARC galaxy outskirts
def sparc_outskirts(folder):
    print("\n=== B. Galaxy outskirts (SPARC) ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fit_a0 import load, accelerations
    names, arr = load(folder)
    gobs, gbar, fe = accelerations(arr)
    k = (fe < 0.10) & (gbar > 0)
    gobs, gbar, names = gobs[k], gbar[k], names[k]

    def rms(a0, e):
        pred = g_rule13(gbar, e * a0, a0)
        return np.sqrt(np.mean((np.log10(gobs) - np.log10(pred)) ** 2))

    a0s = np.linspace(0.8e-10, 1.6e-10, 81)
    es = np.concatenate([[0.0], np.linspace(0.005, 0.3, 60)])
    grid = np.array([[rms(a, e) for e in es] for a in a0s])
    ia, ie = np.unravel_index(np.argmin(grid), grid.shape)
    best0 = grid[:, 0].min()
    a_best0 = a0s[np.argmin(grid[:, 0])]
    print(f"No outside pull:        a0 = {a_best0:.3e}, mismatch {best0:.4f} dex")
    print(f"Best single outside pull: e = {es[ie]:.3f} a0, a0 = {a0s[ia]:.3e}, mismatch {grid[ia, ie]:.4f} dex")
    for e in (0.001, 0.003, 0.01, 0.02, 0.05):
        m = min(rms(a, e) for a in a0s)
        print(f"   same outside pull e = {e:5.3f} a0 for every galaxy: best mismatch {m:.4f} dex")

    # residuals vs gbar for e = 0 (does the faint end bend down?)
    r0 = np.log10(gobs) - np.log10(g_rule13(gbar, 0.0, a_best0))
    edges = np.array([1e-13, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 1e-8])
    print("Average leftover (observed vs predicted, dex) by inside pull, no outside pull assumed:")
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gbar >= lo) & (gbar < hi)
        if m.sum() < 10:
            continue
        mean, err = r0[m].mean(), r0[m].std() / math.sqrt(m.sum())
        # what rule 13 with e=0.05 a0 would add at the centre of the bin
        gc = math.sqrt(lo * hi)
        shift = math.log10(g_rule13(gc, 0.05 * a_best0, a_best0) / g_rule13(gc, 0.0, a_best0))
        bins.append((gc / A0, mean, err, m.sum(), shift))
        print(f"  g_bar {lo:.0e}-{hi:.0e}: {m.sum():4d} points  leftover {mean:+.3f} +/- {err:.3f}   "
              f"(rule 13 with e = 0.05 a0 predicts {shift:+.3f})")
    return bins


# ---------------------------------------------------------------- C. Crater II
def crater2():
    print("\n=== C. Crater II (faint satellite of the Milky Way) ===")
    L = 1.6e5
    rh = 1066 * 3.0857e16
    r12 = 4 * rh / 3
    V_mw, D = 184e3, 120 * KPC
    g_ex = V_mw**2 / D
    for ups in (1, 2, 4):
        Mst = ups * L * M_SUN
        s_efe = math.sqrt(A0 * G * Mst / (3 * g_ex * r12))
        s_iso = (4 / 81 * G * Mst * A0) ** 0.25
        print(f"  stars weigh {ups} x light: predicted {s_efe / 1e3:.2f} km/s with Milky Way's pull, "
              f"{s_iso / 1e3:.2f} km/s if isolated")
    print(f"  Milky Way's pull there: {g_ex / A0:.3f} a0;  measured (Caldwell et al. 2017): 2.7 +/- 0.3 km/s")


# ---------------------------------------------------------------- D. Double slit, watched and unwatched
def double_slit(n_photons=200_000, seed=3):
    """Red light, slits 0.1 mm wide and 0.5 mm apart, screen 1 m away.
    'record' = how distinguishable the detector's two possible records are (0 = no record, 1 = perfect).
    The detector's records overlap by gamma = sqrt(1 - record^2), which sets the fringe visibility."""
    print("\n=== D. Photon double slit: how strongly the detector records the path ===")
    lam, d, a, Lscr = 650e-9, 0.5e-3, 0.1e-3, 1.0
    x = np.linspace(-0.01, 0.01, 4001)
    th = x / Lscr
    beta = math.pi * a * np.sin(th) / lam
    env = np.sinc(beta / math.pi) ** 2
    phase = 2 * math.pi * d * np.sin(th) / lam
    rng = np.random.default_rng(seed)
    curves = {}
    print("  record strength D | predicted visibility V | V^2 + D^2 | fringe contrast in 200,000 simulated photons")
    for record in (0.0, 0.5, 0.9, 1.0):
        gamma = math.sqrt(1 - record**2)
        I = env * (1 + gamma * np.cos(phase))
        hits = rng.choice(x, size=n_photons, p=I / I.sum())
        h, edges = np.histogram(hits, bins=100, range=(-0.0013, 0.0013))     # central two fringes
        centres = 0.5 * (edges[1:] + edges[:-1])
        env_c = np.interp(centres, x, env)
        flat = h / env_c                                                        # remove the single-slit envelope
        ph = np.interp(centres, x, phase)
        Amat = np.column_stack([np.ones_like(ph), np.cos(ph), np.sin(ph)])      # fit A + B cos + C sin
        coef, *_ = np.linalg.lstsq(Amat, flat, rcond=None)
        V_meas = math.hypot(coef[1], coef[2]) / coef[0]
        curves[record] = I
        print(f"  {record:17.2f} | {gamma:22.2f} | {gamma**2 + record**2:9.2f} | {V_meas:.2f}")
    return x, curves


# ---------------------------------------------------------------- E. Can presence itself act as the watcher?
def presence_as_watcher():
    print("\n=== E. Can presence itself record which path? ===")
    # photon: presence difference its two paths leave at a detector 1 m away, slits 100 um apart
    E = 6.626e-34 * c / 650e-9
    dP = G * E / c**4 * (1 / 1.0 - 1 / (1.0 + 100e-6))
    print(f"  Red photon: difference in presence its two paths leave 1 m away = {dP:.1e}  (nothing to record)")
    print("  Superposed objects, collapse time if presence differences end superpositions (tau = hbar / E_G):")
    cases = [("Largest molecule interfered so far (~25,000 u, ~1 nm)", 25000 * 1.66054e-27, 1e-9),
             ("Virus-sized particle (~1e-19 kg, 50 nm)", 1e-19, 50e-9),
             ("1-micrometre glass bead", 2200 * 4 / 3 * math.pi * (0.5e-6) ** 3, 0.5e-6),
             ("10-micrometre glass bead", 2200 * 4 / 3 * math.pi * (5e-6) ** 3, 5e-6)]
    for label, m, R in cases:
        EG = 6 / 5 * G * m * m / R
        tau = HBAR / EG
        print(f"    {label:52s} tau = {tau:.2g} s")


if __name__ == "__main__":
    wide_binaries()
    if len(sys.argv) > 1:
        sparc_outskirts(sys.argv[1])
    crater2()
    double_slit()
    presence_as_watcher()

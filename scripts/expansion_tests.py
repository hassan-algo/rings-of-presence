"""
Round 2, test E: what drives the expansion? Fit the Pantheon+ supernovae (Scolnic et al. 2022, Brout et al. 2022)
with full statistical + systematic covariance, comparing:
  1. matter + emptiness (constant presence of empty space, Lambda)          [the surviving theory, rule 8]
  2. matter only (no emptiness)
  3. energized dark matter: ordinary matter + dark matter with pressure (w = 1/3)
  4. open universe, matter only (Omega_m = 0.31, curvature instead of emptiness)
  5. empty coasting universe
  6. tired light: static space, photons lose energy on the way (no time dilation)
  7. matter + 'emptiness' with free tension w (Omega_m fixed at 0.315)
Absolute magnitude M is profiled analytically. Supernovae with z > 0.01.
Usage: python3 expansion_tests.py <folder with Pantheon+SH0ES.dat and Pantheon+SH0ES_STAT+SYS.cov>
"""
import sys, numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
C = 299792.458

def load(folder):
    head = open(f"{folder}/Pantheon+SH0ES.dat").readline().split()
    d = np.genfromtxt(f"{folder}/Pantheon+SH0ES.dat", names=head, dtype=None, encoding=None, skip_header=1)
    raw = np.loadtxt(f"{folder}/Pantheon+SH0ES_STAT+SYS.cov")
    n = int(raw[0]); cov = raw[1:].reshape(n, n)
    keep = (d["zHD"] > 0.01)
    idx = np.where(keep)[0]
    return d["zHD"][idx], d["zHEL"][idx], d["m_b_corr"][idx], cov[np.ix_(idx, idx)]

def comoving(z, E):
    zz = np.linspace(0, z.max() * 1.001, 20001)
    dc = cumulative_trapezoid(1 / E(zz), zz, initial=0)
    return np.interp(z, zz, dc)      # in units of c/H0

def dl_flrw(z, zhel, Om=0.0, Ol=0.0, Ohot=0.0, Ok=0.0, w=-1.0):
    E = lambda x: np.sqrt(Om * (1 + x)**3 + Ohot * (1 + x)**4 + Ol * (1 + x)**(3 * (1 + w)) + Ok * (1 + x)**2)
    dc = comoving(z, E)
    if Ok > 1e-9:
        dm = np.sinh(np.sqrt(Ok) * dc) / np.sqrt(Ok)
    else:
        dm = dc
    return (1 + zhel) * dm            # c/H0 units; H0 absorbed into M

def dl_tired(z, zhel):
    return np.log(1 + z) * np.sqrt(1 + z)

def chi2(mu_shape, m, icov):
    r = m - 5 * np.log10(mu_shape)
    one = np.ones_like(r)
    a = one @ icov @ one; b = one @ icov @ r; cc = r @ icov @ r
    return cc - b * b / a, b / a      # profiled over the additive constant

if __name__ == "__main__":
    z, zhel, m, cov = load(sys.argv[1])
    icov = np.linalg.inv(cov)
    print(f"{len(z)} supernovae, z = {z.min():.3f} to {z.max():.2f}")
    res = {}
    f = lambda Om: chi2(dl_flrw(z, zhel, Om=Om, Ol=1 - Om), m, icov)[0]
    best = minimize_scalar(f, bounds=(0.05, 0.7), method="bounded")
    # 1-sigma on Om
    grid = np.linspace(0.2, 0.45, 251); cg = np.array([f(o) for o in grid])
    ok = grid[cg <= best.fun + 1]
    res["1. matter + emptiness (Lambda)"] = best.fun
    print(f"  best Omega_m = {best.x:.3f} (+{ok.max()-best.x:.3f}/-{best.x-ok.min():.3f}); emptiness share = {1-best.x:.3f}")
    res["2. matter only"] = chi2(dl_flrw(z, zhel, Om=1), m, icov)[0]
    res["3. energized dark matter (w=+1/3)"] = chi2(dl_flrw(z, zhel, Om=0.049, Ohot=0.951), m, icov)[0]
    res["4. open, matter only (Om=0.31)"] = chi2(dl_flrw(z, zhel, Om=0.31, Ok=0.69), m, icov)[0]
    res["5. empty, coasting"] = chi2(dl_flrw(z, zhel, Ok=1.0), m, icov)[0]
    res["6. tired light, static"] = chi2(dl_tired(z, zhel), m, icov)[0]
    fw = lambda w: chi2(dl_flrw(z, zhel, Om=0.315, Ol=0.685, w=w), m, icov)[0]
    bw = minimize_scalar(fw, bounds=(-2.0, 0.5), method="bounded")
    wg = np.linspace(-1.5, -0.5, 201); wc = np.array([fw(x) for x in wg]); wok = wg[wc <= bw.fun + 1]
    res["7. emptiness with free tension w"] = bw.fun
    print(f"  free tension: w = {bw.x:.3f} (+{wok.max()-bw.x:.3f}/-{bw.x-wok.min():.3f}); expansion speeds up only if w < -1/3")
    base = res["1. matter + emptiness (Lambda)"]
    print(f"\n  {'model':38s} {'chi2':>9s} {'worse by':>9s}")
    for k, v in res.items():
        print(f"  {k:38s} {v:9.1f} {v-base:9.1f}")
    # binned residuals relative to the coasting universe, for plotting
    _, off = chi2(dl_flrw(z, zhel, Om=best.x, Ol=1 - best.x), m, icov)
    mu_data = m - off
    ref = 5 * np.log10(dl_flrw(z, zhel, Ok=1.0))
    edges = np.array([0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 2.3])
    print("\n  binned data minus coasting universe (mag): z, value, error, n")
    err = np.sqrt(np.diag(cov))
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (z >= lo) & (z < hi)
        if k.sum() < 3: continue
        wts = 1 / err[k]**2
        val = np.sum(wts * (mu_data[k] - ref[k])) / wts.sum()
        print(f"  {np.median(z[k]):.3f} {val:+.3f} {1/np.sqrt(wts.sum()):.3f} {k.sum()}")
    zz = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0])
    refz = 5 * np.log10(dl_flrw(zz, zz, Ok=1.0))
    print("\n  model curves minus coasting (mag), same offset as data:")
    for lab, dl in [("Lambda", dl_flrw(zz, zz, Om=best.x, Ol=1 - best.x)), ("matter only", dl_flrw(zz, zz, Om=1)),
                    ("energized DM", dl_flrw(zz, zz, Om=0.049, Ohot=0.951)), ("tired light", dl_tired(zz, zz))]:
        print("  " + lab + ": " + " ".join(f"{v:+.3f}" for v in 5 * np.log10(dl) - refz))

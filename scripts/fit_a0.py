"""
Fit MOND's a0 to the SPARC rotation curves (175 galaxies, Lelli, McGaugh & Schombert 2016)
by scanning a0 over a range and keeping the value with the smallest mismatch.
Then compare with values built from the presence model's own ingredients (c, H0, Lambda).

Usage: python3 fit_a0.py <folder with *_rotmod.dat files>
"""
import sys, glob, os
import numpy as np

KMS2_PER_KPC = 1e6 / 3.0856776e19          # (km/s)^2 / kpc  ->  m/s^2
C = 299_792_458.0
MPC = 3.0856776e22


def load(folder):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*_rotmod.dat"))):
        name = os.path.basename(path).replace("_rotmod.dat", "")
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            R, Vobs, eV, Vgas, Vdisk, Vbul = map(float, line.split()[:6])
            rows.append((name, R, Vobs, eV, Vgas, Vdisk, Vbul))
    names = np.array([r[0] for r in rows])
    arr = np.array([r[1:] for r in rows])
    return names, arr


def accelerations(arr, ups_disk=0.5, ups_bul=0.7):
    R, Vobs, eV, Vgas, Vdisk, Vbul = arr.T
    sq = lambda v: v * np.abs(v)                                  # signed square
    Vbar2 = sq(Vgas) + ups_disk * sq(Vdisk) + ups_bul * sq(Vbul)
    gobs = Vobs**2 / R * KMS2_PER_KPC
    gbar = Vbar2 / R * KMS2_PER_KPC
    return gobs, gbar, eV / Vobs


def g_pred(gbar, a0, form="exp"):
    y = gbar / a0
    if form == "exp":                       # nu(y) = 1/(1 - exp(-sqrt(y)))  (McGaugh et al. 2016)
        return gbar / (-np.expm1(-np.sqrt(y)))
    else:                                   # "simple": mu(x) = x/(1+x)
        return gbar * (0.5 + np.sqrt(0.25 + 1 / y))


def mismatch(gobs, gbar, a0, form="exp"):
    """RMS of log10 residuals, in dex: every data point counts equally."""
    r = np.log10(gobs) - np.log10(g_pred(gbar, a0, form))
    return np.sqrt(np.mean(r**2))


def scan(gobs, gbar, form="exp", lo=0.2e-10, hi=3.0e-10, n=2801):
    grid = np.linspace(lo, hi, n)
    m = np.array([mismatch(gobs, gbar, a, form) for a in grid])
    return grid, m, grid[np.argmin(m)]


if __name__ == "__main__":
    folder = sys.argv[1]
    names, arr = load(folder)
    print(f"Loaded {len(arr)} points from {len(set(names))} galaxies")

    gobs, gbar, frac_err = accelerations(arr)
    keep = (frac_err < 0.10) & (gbar > 0)               # McGaugh et al. 2016 per-point cut
    names_k, gobs_k, gbar_k = names[keep], gobs[keep], gbar[keep]
    print(f"After cut (velocity error < 10%): {keep.sum()} points, {len(set(names_k))} galaxies")

    # Newton only (no a0) for reference
    newton = np.sqrt(np.mean((np.log10(gobs_k) - np.log10(gbar_k))**2))
    print(f"Mismatch with Newton only (no dark matter, no a0): {newton:.3f} dex")

    grid, m, best = scan(gobs_k, gbar_k)
    print(f"Best a0 (scan): {best:.4e} m/s^2, mismatch {m.min():.3f} dex")
    np.savetxt("a0_scan.csv", np.column_stack([grid, m]), delimiter=",",
               header="a0_m_s2,rms_dex", comments="")

    # Statistical uncertainty: bootstrap over galaxies
    rng = np.random.default_rng(1)
    gal = np.array(sorted(set(names_k)))
    idx_by_gal = {g: np.where(names_k == g)[0] for g in gal}
    fine = np.linspace(best * 0.8, best * 1.2, 401)
    boots = []
    for _ in range(300):
        pick = rng.choice(gal, size=len(gal), replace=True)
        ii = np.concatenate([idx_by_gal[g] for g in pick])
        mm = [mismatch(gobs_k[ii], gbar_k[ii], a) for a in fine]
        boots.append(fine[int(np.argmin(mm))])
    boots = np.array(boots)
    print(f"Bootstrap over galaxies: a0 = {best:.3e} +/- {boots.std():.2e}")

    # Systematic: stellar mass-to-light ratio (the biggest assumption)
    print("Effect of the stellar mass-to-light assumption:")
    for ud in (0.4, 0.5, 0.6):
        go, gb, fe = accelerations(arr, ups_disk=ud, ups_bul=ud + 0.2)
        k = (fe < 0.10) & (gb > 0)
        _, mm, b = scan(go[k], gb[k])
        print(f"  disk {ud:.1f}, bulge {ud + 0.2:.1f}:  a0 = {b:.3e}   mismatch {mm.min():.3f} dex")

    # Other interpolating function
    _, mm, b_simple = scan(gobs_k, gbar_k, form="simple")
    print(f"With the 'simple' transition formula instead: a0 = {b_simple:.3e} (mismatch {mm.min():.3f} dex)")

    # Per-galaxy fits: is one a0 good for all galaxies?
    per = []
    for g in gal:
        ii = idx_by_gal[g]
        if len(ii) < 5:
            continue
        _, mm, b = scan(gobs_k[ii], gbar_k[ii], lo=0.05e-10, hi=6e-10, n=1200)
        per.append(b)
    per = np.array(per)
    q = np.percentile(per, [16, 50, 84])
    print(f"Per-galaxy best a0 ({len(per)} galaxies with >=5 points): median {q[1]:.2e}, "
          f"middle 68% {q[0]:.2e} to {q[2]:.2e}")

    # ----- values built from the model's ingredients
    print("\nCandidates from the model's ingredients:")
    cands = {}
    for label, H0, OL in (("Planck H0=67.4", 67.4, 0.685), ("SH0ES H0=73.0", 73.0, 0.685)):
        H = H0 * 1e3 / MPC
        cands[f"c*H0/2pi ({label})"] = C * H / (2 * np.pi)
        cands[f"c^2*sqrt(Lambda/3)/2pi ({label})"] = C * H * np.sqrt(OL) / (2 * np.pi)
    for k, v in cands.items():
        print(f"  {k:42s} {v:.3e}   ratio to fit {v / best:.2f}   mismatch {mismatch(gobs_k, gbar_k, v):.3f} dex")

    print("\nNumber the model would have to predict (instead of 2pi):")
    for label, H0, OL in (("Planck", 67.4, 0.685), ("SH0ES", 73.0, 0.685)):
        H = H0 * 1e3 / MPC
        print(f"  {label}:  c*H0/a0 = {C * H / best:.2f}    c^2*sqrt(Lambda/3)/a0 = {C * H * np.sqrt(OL) / best:.2f}"
              f"   (2pi = {2 * np.pi:.2f})")

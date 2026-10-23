"""
The fix: one star's motion seen from the other (relative orbit), with every (v/c)^2 and (v/c)^4
piece included exactly so they cancel over each orbit, plus the (v/c)^5 wave-energy term.

Equations of motion in harmonic coordinates (Blanchet, Living Reviews in Relativity 2014, eq. 219-222):
    a = -(G M / r^2) [ (1 + A) n + B v ]
In the presence model's language:
    (v/c)^2 terms  ("1PN") = rules 2b (presence-on-presence) + 7 (clocks & rulers) + 11 (where/when,
                              velocity carried)  -- Einstein, Infeld & Hoffmann 1938
    (v/c)^4 terms  ("2PN") = the same rules, one level deeper
    (v/c)^5 terms ("2.5PN")= rule 9: ripples carrying energy away

Test system: PSR B1913+16 (Hulse-Taylor), real eccentric orbit.
We time each periastron passage, like pulsar astronomers time the real system.
"""
import math, sys, time
import numpy as np
from scipy.integrate import solve_ivp

G = 6.67430e-11
c = 299_792_458.0
GM_SUN = 1.32712440018e20
M_SUN = GM_SUN / G
DAY = 86400.0
YEAR = 365.25 * DAY

m1, m2 = 1.438 * M_SUN, 1.390 * M_SUN        # Weisberg & Huang 2016
M = m1 + m2
GM = G * M
NU = m1 * m2 / M**2
PB = 0.322997448918 * DAY
ECC = 0.6171340
A_KEP = (GM * PB**2 / (4 * math.pi**2)) ** (1 / 3)


def make_rhs(pn1=True, pn2=True, pn25=True):
    c2, c4, c5 = c**2, c**4, c**5
    nu = NU

    def rhs(t, y):
        x, yy, vx, vy = y
        r = math.hypot(x, yy)
        nx, ny = x / r, yy / r
        rd = nx * vx + ny * vy                  # radial speed
        v2 = vx * vx + vy * vy
        u = GM / r
        A = 0.0
        B = 0.0
        if pn1:
            A += (-1.5 * rd * rd * nu + v2 + 3 * nu * v2 - u * (4 + 2 * nu)) / c2
            B += (-4 * rd + 2 * rd * nu) / c2
        if pn2:
            rd2 = rd * rd
            A += (15 / 8 * rd2**2 * nu - 45 / 8 * rd2**2 * nu**2 - 4.5 * rd2 * nu * v2
                  + 6 * rd2 * nu**2 * v2 + 3 * nu * v2**2 - 4 * nu**2 * v2**2
                  + u * (-2 * rd2 - 25 * rd2 * nu - 2 * rd2 * nu**2 - 6.5 * nu * v2 + 2 * nu**2 * v2)
                  + u * u * (9 + 87 / 4 * nu)) / c4
            B += (4.5 * rd2 * rd * nu + 3 * rd2 * rd * nu**2 - 7.5 * rd * nu * v2 - 2 * rd * nu**2 * v2
                  + u * (2 * rd + 20.5 * rd * nu + 4 * rd * nu**2)) / c4
        if pn25:
            A += (-24 / 5 * nu * rd * v2 * u - 136 / 15 * nu * rd * u * u) / c5
            B += (8 / 5 * nu * v2 * u + 24 / 5 * nu * u * u) / c5
        k = -u / r
        return (vx, vy, k * ((1 + A) * nx + B * vx), k * ((1 + A) * ny + B * vy))
    return rhs


def run(n_orbits, rtol=1e-13, **switches):
    rp = A_KEP * (1 - ECC)
    vp = math.sqrt(GM * (1 + ECC) / (A_KEP * (1 - ECC)))
    y0 = [rp, 0.0, 0.0, vp]

    def periastron(t, y):                       # radial speed crosses zero going outward
        return y[0] * y[2] + y[1] * y[3]
    periastron.direction = 1

    sol = solve_ivp(make_rhs(**switches), (0, (n_orbits + 0.5) * PB), y0, method="DOP853",
                    rtol=rtol, atol=[1e-6, 1e-6, 1e-10, 1e-10], events=periastron)
    te = sol.t_events[0]
    ye = sol.y_events[0]
    te, ye = te[te > 0.25 * PB], ye[te > 0.25 * PB]          # skip the start point
    return te, ye


def analyse(te, ye):
    n = np.arange(len(te))
    # Fit t_n = t0 + P n + (1/2) P Pdot n^2  (exactly how pulsar timing measures Pdot)
    nc = n - n.mean()
    coef = np.polyfit(nc, te - te.mean(), 2)
    P = coef[1]
    pdot = 2 * coef[0] / P
    resid = te - te.mean() - np.polyval(coef, nc)
    # Periastron direction -> advance per year
    ang = np.unwrap(np.arctan2(ye[:, 1], ye[:, 0]))
    adv = np.polyfit(te, ang, 1)[0] * YEAR * 180 / math.pi
    return P, pdot, adv, np.abs(resid).max()


def peters_mathews(pb, e):
    f = (1 + 73 / 24 * e**2 + 37 / 96 * e**4) / (1 - e**2) ** 3.5
    return (-192 * math.pi * G ** (5 / 3) / (5 * c**5) * (pb / (2 * math.pi)) ** (-5 / 3)
            * f * m1 * m2 * M ** (-1 / 3))


def advance_1pn(pb, e):
    return 3 * (2 * math.pi / pb) ** (5 / 3) * GM ** (2 / 3) / (c**2 * (1 - e**2)) * YEAR * 180 / math.pi


if __name__ == "__main__":
    n_orb = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print(f"Hulse-Taylor orbit: Pb = {PB / 3600:.3f} h, e = {ECC}, a = {A_KEP / 1e3:,.0f} km, {n_orb} orbits "
          f"({n_orb * PB / DAY:.0f} days)\n")
    cases = [
        ("Newton only",                     dict(pn1=False, pn2=False, pn25=False)),
        ("Fixed rules, no waves (control)", dict(pn1=True,  pn2=True,  pn25=False)),
        ("Fixed rules + waves (rule 9)",    dict(pn1=True,  pn2=True,  pn25=True)),
    ]
    res = {}
    for label, sw in cases:
        t0 = time.time()
        te, ye = run(n_orb, **sw)
        P, pdot, adv, rmax = analyse(te, ye)
        res[label] = (P, pdot, adv)
        da = 2 / 3 * A_KEP * pdot          # change in separation per orbit
        print(f"{label:33s} passages {len(te):4d}  period {P:.6f} s  dPb/dt {pdot:+.4e}  "
              f"({da * 1e3:+.3f} mm/orbit)  periastron advance {adv:.4f} deg/yr  "
              f"[fit residual {rmax:.1e} s, {time.time() - t0:.0f}s]")

    pm = peters_mathews(PB, ECC)
    print(f"\nPredicted wave decay (Peters & Mathews 1963): dPb/dt = {pm:.4e}  "
          f"({2 / 3 * A_KEP * pm * 1e3:.3f} mm/orbit)")
    sim = res["Fixed rules + waves (rule 9)"][1] - res["Fixed rules, no waves (control)"][1]
    print(f"Simulated (full minus control):              dPb/dt = {sim:.4e}   ratio to prediction {sim / pm:.4f}")
    print(f"Measured / predicted for the real pulsar (Weisberg & Huang 2016): 0.9983 +/- 0.0016")
    print(f"\nPeriastron advance: (v/c)^2 formula {advance_1pn(PB, ECC):.4f} deg/yr;  measured 4.2266 deg/yr")

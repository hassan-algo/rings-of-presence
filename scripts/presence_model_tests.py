"""
Testing the ring/presence model against real experiments.

Presence P = G*E / (c^4 * r)  (dimensionless; equals GM/(c^2 r) for matter).
Rule variants are switched on/off to show which rule does the work.

Rules referenced:
  1  presence formula + ring rule (grows at c)
  2b presence dims itself within one ring   -> PPN beta  (1 = on, 0 = off)
  3  old: a = c^2 grad P (Newton);  new: most-time-on-own-clock path
  4  photons carry/create presence
  5  photon energy shifts by P_arrive - P_emit (clock half)
  7  rulers stretch by (1+P)                -> PPN gamma (1 = on, 0 = off)
"""
import numpy as np
from scipy.integrate import solve_ivp

G = 6.67430e-11
c = 299_792_458.0
GM_SUN = 1.32712440018e20          # m^3 s^-2
R_SUN = 6.957e8                    # m
M_SUN = GM_SUN / G
L_SUN = 3.828e26                   # W
GM_EARTH = 3.986004418e14
R_EARTH = 6.371e6
AU = 1.495978707e11
YEAR = 365.25 * 86400
DAY = 86400.0
RAD2ARCSEC = 180 / np.pi * 3600

out = {}

# ---------------------------------------------------------------- 1. Pound-Rebka
h = 22.5
g = GM_EARTH / R_EARTH**2
out["pound_rebka"] = g * h / c**2          # rule 5: dE/E = P_arrive - P_emit

# ---------------------------------------------------------------- 2. GPS clocks
r_gps = 26_560e3
P_ground = GM_EARTH / (c**2 * R_EARTH)
P_orbit = GM_EARTH / (c**2 * r_gps)
grav = (P_ground - P_orbit) * DAY * 1e6            # us/day, satellite clock runs fast
v_gps = np.sqrt(GM_EARTH / r_gps)
vel = -(v_gps**2 / (2 * c**2)) * DAY * 1e6         # us/day, motion slows it
out["gps"] = (grav, vel, grav + vel)

# ---------------------------------------------------------------- 3. Light bending
def bending(gamma, b=R_SUN):
    return (1 + gamma) / 2 * 4 * GM_SUN / (c**2 * b) * RAD2ARCSEC
out["bend_old"] = bending(0)    # rules 1+3 only
out["bend_new"] = bending(1)    # rules 3+7

# ---------------------------------------------------------------- 4. Radar delay (Venus, superior conjunction, grazing Sun)
def shapiro_roundtrip(gamma, r1=AU, r2=0.723 * AU, b=R_SUN):
    return 2 * (1 + gamma) * GM_SUN / c**3 * np.log(4 * r1 * r2 / b**2)
out["shapiro_old"] = shapiro_roundtrip(0) * 1e6
out["shapiro_new"] = shapiro_roundtrip(1) * 1e6

# ---------------------------------------------------------------- 5. Mercury, integrated numerically
a_m, e_m = 5.7909050e10, 0.205630
T_m = 2 * np.pi * np.sqrt(a_m**3 / GM_SUN)

def mercury_shift(pn_on, beta, gamma, n_orbits=60):
    mu = GM_SUN
    def rhs(t, s):
        x, y, vx, vy = s
        r = np.hypot(x, y)
        nx, ny = x / r, y / r
        ax, ay = -mu / r**2 * nx, -mu / r**2 * ny
        if pn_on:
            v2 = vx * vx + vy * vy
            ndotv = nx * vx + ny * vy
            k = mu / (c**2 * r**2)
            radial = 2 * (beta + gamma) * mu / r - gamma * v2
            ax += k * (radial * nx + 2 * (1 + gamma) * ndotv * vx)
            ay += k * (radial * ny + 2 * (1 + gamma) * ndotv * vy)
        return [vx, vy, ax, ay]

    rp = a_m * (1 - e_m)
    vp = np.sqrt(mu * (1 + e_m) / (a_m * (1 - e_m)))
    def peri(t, s):                      # r.v = 0 going from - to + => perihelion
        return s[0] * s[2] + s[1] * s[3]
    peri.direction = 1
    sol = solve_ivp(rhs, (0, n_orbits * T_m), [rp, 0, 0, vp], method="DOP853",
                    rtol=1e-13, atol=1e-6, events=peri)
    te, ye = sol.t_events[0], sol.y_events[0]
    ang = np.unwrap(np.arctan2(ye[:, 1], ye[:, 0]))
    slope = np.polyfit(te, ang, 1)[0]                 # rad per second
    return slope * 100 * YEAR * RAD2ARCSEC            # arcsec per century

def mercury_analytic(beta, gamma):
    per_orbit = 6 * np.pi * GM_SUN / (c**2 * a_m * (1 - e_m**2)) * (2 + 2 * gamma - beta) / 3
    return per_orbit * (100 * YEAR / T_m) * RAD2ARCSEC

variants = {
    "Old rule 3 (pull only)":                 (False, 0, 0),
    "New rule 3, clocks only":                (True, 0, 0),
    "New rule 3 + 2b (no rule 7)":            (True, 1, 0),
    "New rule 3 + rule 7 (no 2b)":            (True, 0, 1),
    "Full model: rules 2b + 3 + 7":           (True, 1, 1),
}
out["mercury"] = {k: (mercury_shift(*v), mercury_analytic(v[1], v[2]) if v[0] else 0.0)
                  for k, v in variants.items()}

# ---------------------------------------------------------------- 6. Sun getting lighter (rule 4)
light_rate = -(L_SUN / c**2) * YEAR / M_SUN      # fraction per year
out["sun_light"] = light_rate
out["sun_total"] = (light_rate - 0.2e-13, light_rate - 0.69e-13)   # + solar wind range

# ---------------------------------------------------------------- 7. GW170817 (ring rule)
travel = 130e6 * YEAR                       # ~130 million light-years, in seconds of light travel
out["gw"] = 1.7 / travel                    # naive fractional speed difference if all delay were speed

# ---------------------------------------------------------------- 8. 5th dimension (Gauss's law on R^3 x circle)
# Potential of a point mass with one compact dimension of radius R (sum over images):
#   phi(r) ∝ (1/r) * coth(r / 2R)  -> Newton*(1 + 2 e^{-r/R}) far away, 1/r^2 up close
def force_ratio(r, R):
    x = r / (2 * R)
    coth = 1 / np.tanh(x)
    csch2 = 1 / np.sinh(x) ** 2
    # F ∝ -d/dr[(1/r) coth(r/2R)] = coth/r^2 + csch^2/(2R r);  normalise to Newton 1/r^2
    return coth + csch2 * r / (2 * R)
out["fifth"] = {R: force_ratio(52e-6, R * 1e-6) for R in (100, 50, 30, 10)}
out["fifth_alpha"] = 2.0   # far-field Yukawa strength of this model, lambda = R

# ---------------------------------------------------------------- 9. a0 tracking the expansion (new direction)
H0_planck, H0_shoes = 67.4, 73.0
def a0_from_H(H0_kms_mpc):
    H = H0_kms_mpc * 1e3 / 3.0857e22
    return c * H / (2 * np.pi)
out["a0_today"] = (a0_from_H(H0_planck), a0_from_H(H0_shoes))
Om, OL = 0.315, 0.685
E = lambda z: np.sqrt(Om * (1 + z) ** 3 + OL)

# Genzel et al. 2017 galaxies as tabulated by Milgrom (2017):
# name, z, x_half (= g/a0 at R_1/2 with local a0), measured dark fraction, upper limit or error
genzel = [
    ("COS4 01351", 0.854, 2.8, "0.21 ± 0.10"),
    ("D3a 6397",   1.500, 3.5, "0.17 (< 0.38)"),
    ("GS4 43501",  1.613, 3.6, "0.19 ± 0.09"),
    ("zC 406690*", 2.196, 4.4, "0 (< 0.08)"),
    ("zC 400569",  2.242, 10.8, "0 (< 0.07)"),
    ("D3a 15504",  2.383, 4.0, "0.12 (< 0.26)"),
]
from scipy.optimize import brentq

def zeta_a(x):
    """Dark ('phantom') fraction with mu(x) = x/(1+x), x = g/a0."""
    return 1 / (1 + x)

def zeta_b(x):
    """Dark fraction with nu(y) = 1/(1 - exp(-sqrt(y))), where x = y*nu(y)."""
    y = brentq(lambda y: y / (1 - np.exp(-np.sqrt(y))) - x, 1e-9, 1e3)
    return np.exp(-np.sqrt(y))

# Milgrom's published constant-a0 values (columns zeta_a, zeta_b) -- used to check this code
milgrom_check = [(0.26, 0.22), (0.22, 0.18), (0.22, 0.17), (0.18, 0.14), (0.08, 0.04), (0.20, 0.16)]

rows = []
for (name, z, x, meas), (pa, pb) in zip(genzel, milgrom_check):
    f = E(z)                          # a0 scaled with the expansion rate H(z)/H0
    rows.append(dict(name=name, z=z, f=f,
                     const_a=zeta_a(x), const_b=zeta_b(x),
                     H_a=zeta_a(x / f), H_b=zeta_b(x / f),
                     pub_a=pa, pub_b=pb, meas=meas))
out["a0_rows"] = rows

# ---------------------------------------------------------------- 10. Light pulling light (new direction)
# Moving-lens deflection: alpha = 4GM/(c^2 b) * gamma * (1 - v_par/c).
# For a pulse of energy E = gamma M c^2 going opposite to the test ray (v_par -> -c):
#   alpha = 8 G E / (c^4 b)   (twice a static mass of the same energy; zero for parallel beams)
def light_on_light(E_J, b):
    return 8 * G * E_J / (c**4 * b)
out["lol_nif"] = light_on_light(2.0e6, 1e-3)     # ~2 MJ (National Ignition Facility class), 1 mm away
out["lol_pw"] = light_on_light(30.0, 1e-5)       # petawatt pulse ~30 J, 10 um away

# ---------------------------------------------------------------- print
if __name__ == "__main__":
    print(f"Pound-Rebka shift: {out['pound_rebka']:.3e}")
    g_, v_, n_ = out["gps"]
    print(f"GPS: gravity {g_:+.1f} us/day, motion {v_:+.1f} us/day, net {n_:+.1f} us/day")
    print(f"Light bending: old {out['bend_old']:.3f}\"  new {out['bend_new']:.3f}\"")
    print(f"Radar delay: old {out['shapiro_old']:.0f} us  new {out['shapiro_new']:.0f} us")
    print("Mercury (numerical, analytic) arcsec/century:")
    for k, (num, ana) in out["mercury"].items():
        print(f"  {k:36s} {num:7.2f}  {ana:7.2f}")
    print(f"Sun mass loss from light: {out['sun_light']:.3e}/yr; with wind {out['sun_total'][0]:.3e} to {out['sun_total'][1]:.3e}")
    print(f"GW170817 naive speed difference bound: {out['gw']:.1e}")
    print("5th dimension: force ratio at 52 um for R (um):", {k: round(v, 2) for k, v in out["fifth"].items()})
    print(f"a0 from cH0/2pi: Planck {out['a0_today'][0]:.3e}, SH0ES {out['a0_today'][1]:.3e}  (measured 1.2e-10)")
    print("  galaxy        z     a0 x   const(a,b)   [published a,b]   a0~H(a,b)    measured")
    for r in out["a0_rows"]:
        print(f"  {r['name']:12s} {r['z']:.2f}  {r['f']:.2f}   {r['const_a']:.2f},{r['const_b']:.2f}"
              f"     [{r['pub_a']:.2f},{r['pub_b']:.2f}]       {r['H_a']:.2f},{r['H_b']:.2f}    {r['meas']}")
    print(f"Light on light: NIF {out['lol_nif']:.1e} rad, petawatt {out['lol_pw']:.1e} rad")

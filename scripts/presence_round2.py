"""
Round 2, tests A, C and Q: presence at Earth from every source, clock predictions, three-presence numbers,
and superposition lifetimes (tau = hbar / E_G) for every experiment.
"""
import math
G, c, hbar = 6.674e-11, 2.998e8, 1.0546e-34
MS, ME, MM = 1.989e30, 5.972e24, 7.342e22
AU, RE, dEM = 1.496e11, 6.371e6, 3.844e8
kpc, Mpc = 3.0857e19, 3.0857e22
Lam = 1.1e-52

print("A1. Presence, pull and stretch at Earth from every source")
src = [("Moon", MM, dEM), ("Sun", MS, AU), ("Venus (closest)", 4.867e24, 0.28 * AU), ("Jupiter (closest)", 1.898e27, 4.2 * AU),
       ("Andromeda", 1.5e12 * MS, 765 * kpc), ("Virgo cluster", 8e14 * MS, 16.5 * Mpc)]
print(f"  {'source':18s} {'presence':>10s} {'pull m/s2':>10s} {'stretch m/s2':>12s}")
print(f"  {'Earth itself':18s} {G*ME/(c*c*RE):10.2e} {G*ME/RE**2:10.2e} {'-':>12s}")
for n, M, r in src:
    print(f"  {n:18s} {G*M/(c*c*r):10.2e} {G*M/r**2:10.2e} {2*G*M*RE/r**3:12.2e}")
vc, R0 = 230e3, 8.2 * kpc
print(f"  {'Milky Way':18s} {0.5*(550e3)**2/c**2:10.2e} {vc*vc/R0:10.2e} {(vc/R0)**2*RE:12.2e}")
print(f"  {'whole universe':18s} {0.75:10.2f} {'0 (uniform)':>10s} {Lam*c*c/3*RE:12.2e}  (emptiness)")
print(f"  tides: Sun/Moon = {(MS/AU**3)/(MM/dEM**3):.3f}  (measured equilibrium S2/M2 = 0.46)")

print("\nA2. Clocks")
gm = G * ME
grav = gm / c**2 * (1 / RE - 1 / 2.6571e7) * 86400e6; vel = gm / 2.6571e7 / (2 * c * c) * 86400e6
print(f"  GPS: +{grav:.2f} (presence) -{vel:.2f} (speed) = +{grav-vel:.2f} us/day; built-in offset 4.4647e-10 = {4.4647e-10*86400e6:.2f} us/day")
print(f"  Gravity Probe A, 10,000 km up: presence difference {gm/c**2*(1/RE-1/(RE+1e7)):.3e}; confirmed to 1.4e-4 (1976)")
for e, a in ((0.157, 2.7977e7), (0.162, 2.7977e7)):
    print(f"  Galileo 5/6 (e={e}): periodic clock swing amplitude {2*math.sqrt(gm*a)*e/c**2*1e9:.0f} ns; confirmed to 2.5e-5 (2018)")
print(f"  Pound-Rebka 22.5 m: {9.81*22.5/c**2:.3e}; Bothwell 1 mm: {9.80*1e-3/c**2:.3e} (measured (0.98+/-0.23)e-19)")
print(f"  Earth's yearly clock swing from the Sun's presence: {2*math.sqrt(G*MS*AU)*0.0167/c**2*1e3:.3f} ms (pulsar timing: ~1.6 ms)")

print("\nC. Three presences")
print(f"  Earth-Moon balance point: {dEM*MM/(ME+MM)/1e3:,.0f} km from Earth's centre")
print(f"  Earth's ring (L1/L2): {AU*(ME/(3*MS))**(1/3)/1e9:.2f} million km (SOHO at L1, Webb at L2 ~1.5 million km)")
print(f"  Moon: Sun's pull {G*MS/AU**2:.2e} vs Earth's {G*ME/dEM**2:.2e} -> Moon's path always bends toward the Sun")
print(f"  Moon owned by Earth: Earth's pull / Sun's stretch = {(G*ME/dEM**2)/(2*G*MS*dEM/AU**3):.0f}")
print("  Nordtvedt: Earth self-presence 4.6e-10 vs Moon 1.9e-11; theory (rule 2b applied to pull, fall and inertia alike) -> eta = 0;"
      " lunar laser ranging: eta = (-0.2 +/- 1.1)e-4")

print("\nQ. Superposition lifetimes from presence, tau = hbar / E_G (mass smoothed over >= 0.54 Angstrom)")
R0 = 0.54e-10
def tau_point(m):            # atom-like: separation >> smoothing length
    return hbar / (1.2 * G * m * m / R0)
def tau_ball(m, R):          # ball moved farther than its size
    return hbar / (1.2 * G * m * m / R)
yr = 3.156e7
rows = [("Rb-87 atoms 54 cm apart, ~2 s (Kovachy 2015)", tau_point(87 * 1.6605e-27), 2.0, "interference seen"),
        ("25,000 u molecules (Fein 2019)", tau_ball(25000 * 1.6605e-27, 1e-9), 0.01, "interference seen"),
        ("C70, heated / in gas (Hackermuller 2004, Hornberger 2003)", tau_ball(840 * 1.6605e-27, 0.5e-9), 0.01,
         "lost: photons and gas molecules recorded the path"),
        ("1 um glass bead", tau_ball(2200 * 4 / 3 * math.pi * (0.5e-6)**3, 0.5e-6), None, "not yet done"),
        ("10 um glass bead", tau_ball(2200 * 4 / 3 * math.pi * (5e-6)**3, 5e-6), None, "not yet done")]
for n, t, texp, out in rows:
    ts = f"{t/yr:.2g} years" if t > yr else (f"{t:.2g} s")
    verdict = ("consistent: presence too slow to matter" if texp and t > 100 * texp else ("prediction" if not texp else "TENSION"))
    print(f"  {n:58s} presence lifetime {ts:>14s}; experiment: {out}; {verdict}")
print("  simplest version (mass packed like nuclei): excluded underground at Gran Sasso (Donadi 2020); smoothing >= 0.54 Angstrom required")

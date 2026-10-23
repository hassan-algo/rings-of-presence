"""
Presence judge: one formula, P = sum G*E/(c^4 r) (E = M c^2 -> P = G M/(c^2 r)), read three ways.
  1. presence value P   -> clocks (only differences between two places matter)
  2. pull = slope of P  -> who accelerates you most:  g = G M / r^2
  3. stretch (tide)     -> who owns you: compare the owner's pull with the stretch from
                           everything outside: 2 G M d / D^3 per body, (Lambda c^2/3) d for emptiness
  4. a0 check           -> if ordinary matter's pull < a0, the dark layer's extra pull dominates
"""
import math
G, c = 6.674e-11, 2.998e8
AU, kpc = 1.496e11, 3.0857e19
Lam = 1.1e-52
A0 = 1.13e-10
MSUN = 1.989e30

def judge(name, owner, d, others, galaxy=None):
    """owner = (label, mass); d = distance to owner; others = [(label, mass, distance)]"""
    rows = [(owner[0], owner[1], d)] + others
    print(f"\n=== {name} ===")
    print(f"  {'source':18s} {'presence P':>11s} {'pull m/s^2':>11s}")
    for lab, M, r in rows:
        print(f"  {lab:18s} {G*M/(c*c*r):11.2e} {G*M/r**2:11.2e}")
    if galaxy:
        print(f"  {'Milky Way':18s} {galaxy[0]:11.2e} {galaxy[1]:11.2e}")
    own = G * owner[1] / d**2
    tides = [(lab, 2*G*M*d/r**3) for lab, M, r in others]
    if galaxy:
        tides.append(("Milky Way", galaxy[2] * d))
    tides.append(("emptiness", Lam*c*c/3*d))
    tot = sum(t for _, t in tides)
    print(f"  owner test: {owner[0]}'s pull {own:.2e} vs total stretch from outside {tot:.2e}"
          f"  -> {owner[0]} {'owns it' if own > tot else 'does NOT own it'} ({own/tot:,.0f}x)")
    for lab, t in sorted(tides, key=lambda z: -z[1]):
        print(f"     stretch from {lab:10s} {t:.2e}")

# Milky Way at the Sun: circular speed 230 km/s at 8.2 kpc; potential depth from escape speed ~550 km/s
vc, R0 = 230e3, 8.2*kpc
mw = (0.5*(550e3)**2/c**2, vc*vc/R0, (vc/R0)**2)

judge("The Moon (who owns it?)", ("Earth", 5.972e24), 3.844e8,
      [("Sun", MSUN, AU), ("Jupiter (closest)", 1.898e27, 4.2*AU), ("Venus (closest)", 4.867e24, 0.28*AU)], mw)
judge("The Earth", ("Sun", MSUN), AU,
      [("Jupiter (closest)", 1.898e27, 4.2*AU), ("Venus (closest)", 4.867e24, 0.28*AU)], mw)
judge("Sun's outer comet cloud at 100,000 AU", ("Sun", MSUN), 1e5*AU, [("Alpha Centauri", 2.0*MSUN, 1.34*206265*AU)], mw)
judge("Comet at 400,000 AU", ("Sun", MSUN), 4e5*AU, [("Alpha Centauri", 2.0*MSUN, 1.34*206265*AU)], mw)
print("\nGalaxy pull at the Sun: %.2e m/s^2 = %.1f a0 -> ordinary matter alone ~%.1e (solve a0 rule)" % (vc*vc/R0, vc*vc/R0/A0, 0))
f = lambda gb: gb/(1-math.exp(-math.sqrt(gb/A0)))
from scipy.optimize import brentq
gb = brentq(lambda x: f(x) - vc*vc/R0, 1e-12, 1e-9)
print("  ordinary matter's pull there %.2e (%.1f a0); dark layer supplies %.0f%% of the pull" % (gb, gb/A0, 100*(1-gb/(vc*vc/R0))))

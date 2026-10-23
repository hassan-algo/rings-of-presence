"""
Rule 11 (where and when): presence felt at point x, time t comes from where each object WAS
when its ring set off:  |x - x_j(t_r)| = c (t - t_r).

Test case: two neutron stars with the Hulse-Taylor masses and period (orbit made circular for simplicity).
Each star is pulled by the other's presence, with four choices for WHERE that pull points:

  instant   : at the other star's present position (no delay at all; Newton)
  was       : at where the other star WAS when its presence left (pure delay)
  velocity  : presence also carries the star's velocity, so it points to where the star
              has moved since: x_j(t_r) + v_j(t_r) * (t - t_r)
  vel+acc   : same, also carrying how the velocity is changing:  + a_j(t_r) (t - t_r)^2 / 2

Then: the decay that the remaining lag produces (gravitational waves), against the real pulsar.
"""
import math, sys
import numpy as np

G = 6.67430e-11
c = 299_792_458.0
GM_SUN = 1.32712440018e20
M_SUN = GM_SUN / G
DAY = 86400.0
YEAR = 365.25 * DAY

# Hulse-Taylor pulsar PSR B1913+16 (Weisberg & Huang 2016 masses)
m1, m2 = 1.438 * M_SUN, 1.390 * M_SUN
M = m1 + m2
PB = 0.322997448918 * DAY
ECC = 0.6171340

a0 = (G * M * PB**2 / (4 * math.pi**2)) ** (1 / 3)
om = 2 * math.pi / PB


def circular(t):
    """Exact circular orbit, barycentric frame: positions, velocities, accelerations of both stars."""
    r1, r2 = m2 / M * a0, m1 / M * a0
    cs, sn = math.cos(om * t), math.sin(om * t)
    x1 = (r1 * cs, r1 * sn); v1 = (-om * r1 * sn, om * r1 * cs); A1 = (-om**2 * x1[0], -om**2 * x1[1])
    x2 = (-r2 * cs, -r2 * sn); v2 = (om * r2 * sn, -om * r2 * cs); A2 = (-om**2 * x2[0], -om**2 * x2[1])
    return (x1, x2), (v1, v2), (A1, A2)


def simulate(mode, n_orbits=60, steps_per_orbit=3000):
    dt = PB / steps_per_orbit
    n_pre = int(math.ceil(60.0 / dt)) + 2          # history before t = 0 (the orbit was already going)
    n = n_pre + n_orbits * steps_per_orbit + 1
    X = [[None, None] for _ in range(n)]
    V = [[None, None] for _ in range(n)]
    A = [[None, None] for _ in range(n)]
    t_first = -n_pre * dt
    for k in range(n_pre + 1):
        xs, vs, As = circular(t_first + k * dt)
        for b in (0, 1):
            X[k][b], V[k][b], A[k][b] = xs[b], vs[b], As[b]

    def source(j, tr, last):
        """Position, velocity, acceleration of star j at time tr, from its stored path."""
        u = (tr - t_first) / dt
        k = int(math.floor(u))
        if k >= last:                                   # inside the current step: extrapolate
            d = tr - (t_first + last * dt)
            x, v, a = X[last][j], V[last][j], A[last][j]
            return ((x[0] + v[0] * d + 0.5 * a[0] * d * d, x[1] + v[1] * d + 0.5 * a[1] * d * d),
                    (v[0] + a[0] * d, v[1] + a[1] * d), a)
        s = u - k
        h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
        d00, d10, d01, d11 = 6*s**2 - 6*s, 3*s**2 - 4*s + 1, -6*s**2 + 6*s, 3*s**2 - 2*s
        p0, p1, w0, w1 = X[k][j], X[k + 1][j], V[k][j], V[k + 1][j]
        a_0, a_1 = A[k][j], A[k + 1][j]
        pos = tuple(h00*p0[q] + h10*dt*w0[q] + h01*p1[q] + h11*dt*w1[q] for q in (0, 1))
        vel = tuple((d00*p0[q] + d10*dt*w0[q] + d01*p1[q] + d11*dt*w1[q]) / dt for q in (0, 1))
        acc = tuple((1 - s) * a_0[q] + s * a_1[q] for q in (0, 1))
        return pos, vel, acc

    def accel(t, xs, last):
        out = []
        for i, j, mj in ((0, 1, m2), (1, 0, m1)):
            xi = xs[i]
            if mode == "instant":
                tx, ty = xs[j]
            else:
                tau = math.hypot(xs[j][0] - xi[0], xs[j][1] - xi[1]) / c
                for _ in range(4):                       # find when the arriving ring left
                    p, _, _ = source(j, t - tau, last)
                    tau = math.hypot(p[0] - xi[0], p[1] - xi[1]) / c
                p, v, a = source(j, t - tau, last)
                if mode == "was":
                    tx, ty = p
                elif mode == "velocity":
                    tx, ty = p[0] + v[0] * tau, p[1] + v[1] * tau
                else:  # vel+acc
                    tx = p[0] + v[0] * tau + 0.5 * a[0] * tau * tau
                    ty = p[1] + v[1] * tau + 0.5 * a[1] * tau * tau
            dx, dy = tx - xi[0], ty - xi[1]
            r3 = math.hypot(dx, dy) ** 3
            out.append((G * mj * dx / r3, G * mj * dy / r3))
        return out

    def semi_major(xs, vs):
        r = math.hypot(xs[0][0] - xs[1][0], xs[0][1] - xs[1][1])
        ke = 0.5 * m1 * (vs[0][0]**2 + vs[0][1]**2) + 0.5 * m2 * (vs[1][0]**2 + vs[1][1]**2)
        E = ke - G * m1 * m2 / r
        return -G * m1 * m2 / (2 * E)

    hist = [semi_major(X[n_pre], V[n_pre])]
    last = n_pre
    t = 0.0
    add = lambda P_, Q, f: tuple((P_[b][0] + f * Q[b][0], P_[b][1] + f * Q[b][1]) for b in (0, 1))
    for step in range(n_orbits * steps_per_orbit):
        x0, v0 = X[last], V[last]
        k1v = accel(t, x0, last); k1x = v0
        x2 = add(x0, k1x, dt / 2); v2 = add(v0, k1v, dt / 2)
        k2v = accel(t + dt / 2, x2, last); k2x = v2
        x3 = add(x0, k2x, dt / 2); v3 = add(v0, k2v, dt / 2)
        k3v = accel(t + dt / 2, x3, last); k3x = v3
        x4 = add(x0, k3x, dt); v4 = add(v0, k3v, dt)
        k4v = accel(t + dt, x4, last); k4x = v4
        xn = tuple((x0[b][0] + dt / 6 * (k1x[b][0] + 2*k2x[b][0] + 2*k3x[b][0] + k4x[b][0]),
                    x0[b][1] + dt / 6 * (k1x[b][1] + 2*k2x[b][1] + 2*k3x[b][1] + k4x[b][1])) for b in (0, 1))
        vn = tuple((v0[b][0] + dt / 6 * (k1v[b][0] + 2*k2v[b][0] + 2*k3v[b][0] + k4v[b][0]),
                    v0[b][1] + dt / 6 * (k1v[b][1] + 2*k2v[b][1] + 2*k3v[b][1] + k4v[b][1])) for b in (0, 1))
        t += dt
        last += 1
        X[last] = list(xn); V[last] = list(vn)
        A[last] = accel(t, xn, last - 1) if mode != "instant" else accel(t, xn, last)
        if (step + 1) % steps_per_orbit == 0:
            hist.append(semi_major(xn, vn))
    return np.array(hist)


def gw_period_decay(pb, e, ma, mb):
    """Orbital period change from gravitational waves (Peters & Mathews 1963), dimensionless dPb/dt."""
    f = (1 + 73 / 24 * e**2 + 37 / 96 * e**4) / (1 - e**2) ** 3.5
    return (-192 * math.pi * G ** (5 / 3) / (5 * c**5) * (pb / (2 * math.pi)) ** (-5 / 3)
            * f * ma * mb * (ma + mb) ** (-1 / 3))


if __name__ == "__main__":
    n_orb = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    v_rel = om * a0
    print(f"Binary: separation {a0 / 1e3:,.0f} km, light-time across {a0 / c:.2f} s, "
          f"relative speed {v_rel / 1e3:.0f} km/s (v/c = {v_rel / c:.2e})")
    print(f"Each star moves {m2 / M * v_rel * a0 / c / 1e3:.0f} km while the other's presence crosses the gap\n")

    results = {}
    for mode in ("instant", "was", "velocity", "vel+acc"):
        h = simulate(mode, n_orbits=n_orb)
        results[mode] = h
        days = n_orb * PB / DAY
        print(f"{mode:9s}: separation after {n_orb} orbits ({days:.1f} days): "
              f"{(h[-1] / h[0] - 1) * 100:+.4f}%   ({(h[-1] - h[0]) / 1e3:+,.2f} km)")

    np.savetxt("binary_separation.csv",
               np.column_stack([np.arange(n_orb + 1)] + [results[m] / results[m][0] for m in results]),
               delimiter=",", header="orbit," + ",".join(results), comments="")

    # pure-delay growth rate, analytic: d ln a / dt = 4 v1 v2 / (c a)
    v1, v2 = m2 / M * v_rel, m1 / M * v_rel
    rate = 4 * v1 * v2 / (c * a0)
    print(f"\nPure delay, analytic: separation grows by e (x2.72) every {1 / rate / DAY:.0f} days")

    # Remaining lag -> gravitational waves, real eccentric orbit
    pdot = gw_period_decay(PB, ECC, m1, m2)
    da_dt = (2 / 3) * a0 * pdot / PB
    print(f"\nGravitational-wave decay predicted for PSR B1913+16: dPb/dt = {pdot:.4e}")
    print(f"  = orbit period shorter by {-pdot * YEAR * 1e6:.1f} microseconds per year")
    print(f"  = separation shrinking by {-da_dt * YEAR:.2f} m per year")
    print(f"  Measured / predicted (Weisberg & Huang 2016): 0.9983 +/- 0.0016")
    print(f"  Time to merge from now: {(a0**4 / (4 * 64 / 5 * G**3 * m1 * m2 * M / c**5)) / YEAR / 1e6:.0f} Myr "
          f"(circular-orbit formula; eccentricity makes it shorter)")

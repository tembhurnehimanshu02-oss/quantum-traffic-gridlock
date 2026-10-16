from itertools import combinations
import numpy as np
from traffic import TrafficModel
from qubo import Qubo
j = np.arange(256)[:, None] >> np.arange(8) & 1
k = 1 - 2 * j
f = list(combinations(range(8), 2))
a = np.column_stack([np.ones(256), k] + [k[:, yr] * k[:, zx] for yr, zx in f])
y = [''.join(map(str, cg)) for cg in j]
e = {'uniform': 'Uniform quadratic projection (current baseline)', 'demand': 'Product-weighted projection using cumulative arrival balance', 'greedy_local': 'Projection weighted toward the greedy schedule, with global coverage', 'taylor': 'Second-order discrete Taylor expansion around the greedy schedule', 'boltzmann_1': 'Cost-weighted projection, temperature 1.0 cost standard deviations', 'boltzmann_035': 'Cost-weighted projection, temperature 0.35 cost standard deviations', 'boltzmann_015': 'Cost-weighted projection, temperature 0.15 cost standard deviations', 'self_refine': 'Three rounds weighted by the preceding quadratic surrogate', 'upper_fit': 'Asymmetric regression penalizing cost underprediction fourfold'}

def traffic_costs(scenario):
    zg = TrafficModel(scenario)
    return np.array([zg.evaluate(xe)['cost'] for xe in y])

def fit(xq, ce):
    be = np.sqrt(ce / np.mean(ce))
    return np.linalg.lstsq(a * be[:, None], xq * be, rcond=None)[0]

def low_cost_weights(xq, bv):
    bh = -(xq - xq.mean()) / max(float(xq.std()), 1.0) / bv
    ce = np.exp(bh - bh.max())
    return 0.02 + 0.98 * ce / ce.mean()

def from_coefficients(xm):
    h = xm[1:9].copy()
    d = np.zeros((8, 8))
    for (yr, zx), value in zip(f, xm[9:]):
        d[yr, zx] = value
    g = 4 * d
    np.fill_diagonal(g, -2 * h - 2 * (d.sum(axis=0) + d.sum(axis=1)))
    return Qubo(float(xm[0] + h.sum() + d.sum()), g, float(xm[0]), h, d)

def build(scenario, zk='uniform', xq=None):
    if zk not in e:
        raise ValueError(zk)
    xi = traffic_costs(scenario) if xq is None else np.asarray(xq)
    zn = TrafficModel(scenario)
    xm = a.T @ xi / 256
    if zk == 'demand':
        xx = np.asarray(zn.arrivals, dtype=float).copy()
        ae = zn.zd
        xx[2 + ae, 1:] += 0.5 * np.asarray(zn.transfers)[np.minimum(np.cumsum(xx[ae])[:-1], zn.capacity).astype(int)]
        aj = []
        for zq in range(2):
            zs, ye = np.cumsum(xx[2 * zq:2 * zq + 2], axis=1)
            aj.extend(np.clip((zs + 1) / (zs + ye + 2), 0.2, 0.8))
        ab = np.asarray(aj)
        ce = np.prod(np.where(j, ab, 1 - ab), axis=1)
        xm = fit(xi, 0.1 + 0.9 * ce / ce.mean())
    elif zk == 'greedy_local':
        u = np.array(list(map(int, zn.greedy())))
        ya = np.sum(j != u, axis=1)
        ce = 0.75 ** (8 - ya) * 0.25 ** ya
        xm = fit(xi, 0.1 + 0.9 * ce / ce.mean())
    elif zk == 'taylor':
        u = int(zn.greedy()[::-1], 2)
        xs = j != j[u]
        ag = np.full(256, xi[u])
        for yr in range(8):
            ag += (xi[u ^ 1 << yr] - xi[u]) * xs[:, yr]
        for yr, zx in f:
            bw = xi[u ^ 1 << yr ^ 1 << zx] - xi[u ^ 1 << yr] - xi[u ^ 1 << zx] + xi[u]
            ag += bw * xs[:, yr] * xs[:, zx]
        xm = a.T @ ag / 256
    elif zk.startswith('boltzmann'):
        bv = {'boltzmann_1': 1.0, 'boltzmann_035': 0.35, 'boltzmann_015': 0.15}[zk]
        xm = fit(xi, low_cost_weights(xi, bv))
    elif zk == 'self_refine':
        for l in range(3):
            xm = fit(xi, low_cost_weights(a @ xm, 0.35))
    elif zk == 'upper_fit':
        for l in range(100):
            cb = fit(xi, np.where(a @ xm < xi, 4.0, 1.0))
            if np.linalg.norm(cb - xm) < 1e-10:
                xm = cb
                break
            xm = cb
    return from_coefficients(xm)

def describe(aq, zk):
    aw = aq.to_dict()
    aw.update(model=zk, construction=e[zk], derivation='METHODS.md', conditional_expectations=None, traffic_histories=256, optimal_schedule_used_in_construction=False)
    return aw

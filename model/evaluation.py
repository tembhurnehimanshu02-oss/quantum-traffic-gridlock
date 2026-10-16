from itertools import product
import numpy as np
from scorer import evaluate
from traffic import TrafficModel

def canonical_from_index(ys):
    return ''.join((str(int(ys) >> yr & 1) for yr in range(8)))

def score_table(scenario):
    return {format(zy, '08b'): evaluate(scenario, format(zy, '08b')) for zy in range(256)}

def assess(scenario, aq, bx, yf=None):
    bu = score_table(scenario)
    av = sorted(bu, key=lambda xe: (bu[xe]['cost'], xe))
    zw = bu[av[0]]['cost']
    au = float(np.mean([cc['cost'] for cc in bu.values()]))
    xj = {aw[::-1]: zo for aw, zo in bx.items()}
    bm = sum(xj.values())
    if bm != 4096:
        raise ValueError('Exactly 4096 final shots are required')
    zi = sum((bu[xe]['cost'] * zo for xe, zo in xj.items())) / bm
    yb = av[:13]
    yc = sum((xj.get(xe, 0) for xe in yb)) / bm
    yx = au - zw
    zc = (au - zi) / yx if yx > 1e-12 else None
    zu = sorted(xj, key=lambda xe: (bu[xe]['cost'], xe))
    xa = zu[0]
    zm = min(xj, key=lambda xe: (-xj[xe], xe))
    r = [''.join(map(str, m + xz)) for m, xz in product(((1, 0, 1, 0), (0, 1, 0, 1)), repeat=2)]
    s = min(r, key=lambda xe: (bu[xe]['cost'], xe))
    greedy = TrafficModel(scenario).greedy()
    energies = aq.energies()
    ap = float(energies.min())
    ao = sorted((canonical_from_index(zy) for zy in range(256) if abs(energies[zy] - ap) < 1e-08))
    am = ao[0]

    def record(xe):
        return {'bits': xe, **bu[xe]}
    bp = {**record(am), 'qubo_energy': ap, 'all_minimizers': ao, 'official_costs_of_all_minimizers': {xz: bu[xz]['cost'] for xz in ao}, 'absolute_regret': bu[am]['cost'] - zw, 'relative_regret': (bu[am]['cost'] - zw) / zw if zw else None}
    bc = {'baselines': {'best_alternating': record(s), 'greedy': record(greedy), 'exact_optimum': {**record(av[0]), 'all_minimizers': [xz for xz in av if bu[xz]['cost'] == zw]}, 'uniform_random_mean': au, 'qubo_exact_minimizer': bp}, 'qaoa': {'shots': bm, 'mean_official_cost': zi, 'lift': zc, 'elite_enrichment': yc / (13 / 256), 'elite_fraction': yc, 'elite_set': yb, 'best_observed': record(xa), 'mode': record(zm), 'optimal_shot_fraction': sum((zo for xz, zo in xj.items() if bu[xz]['cost'] == zw)) / bm}, 'degenerate_lift_denominator': yx <= 1e-12}
    if yf is not None:
        ca = np.array([bu[canonical_from_index(zy)]['cost'] for zy in range(256)])
        bc['qaoa']['exact_distribution_mean_official_cost'] = float(yf @ ca)
        bc['qaoa']['exact_distribution_metric_note'] = 'Diagnostic only; headline metrics use 4096 raw shots'
    ah = np.array([energies[int(xe[::-1], 2)] for xe in bu])
    zv = np.array([cc['cost'] for cc in bu.values()])
    bc['surrogate_diagnostics'] = {'uniform_rmse': float(np.sqrt(np.mean((ah - zv) ** 2))), 'max_absolute_error': float(np.max(np.abs(ah - zv))), 'note': 'Quadratic projection is an approximation; exact enumeration here is evaluation only'}
    return (bc, dict(sorted(xj.items())))

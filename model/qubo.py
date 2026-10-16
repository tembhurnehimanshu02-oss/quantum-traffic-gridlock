from dataclasses import dataclass
from itertools import combinations
import numpy as np
from traffic import x, TrafficModel

@dataclass
class Qubo:
    offset: float
    g: np.ndarray
    yw: float
    h: np.ndarray
    d: np.ndarray

    @property
    def scale(self):
        return max(float(np.sqrt(np.sum(self.h ** 2) + np.sum(self.d ** 2))), 1.0)

    def energies(self):
        cg = (np.arange(256)[:, None] >> np.arange(8) & 1).astype(float)
        return self.offset + np.einsum('bi,ij,bj->b', cg, self.g, cg)

    def centered_energies(self):
        ci = 1 - 2 * (np.arange(256)[:, None] >> np.arange(8) & 1)
        return ci @ self.h + np.einsum('bi,ij,bj->b', ci, self.d, ci)

    def to_dict(self):
        return {'model': 'uniform_degree_2_walsh_projection_of_queue_dynamics', 'formula': 'offset + sum_i Q[i][i]*x_i + sum_{i<j} Q[i][j]*x_i*x_j', 'matrix_convention': 'upper triangular; lower triangle is zero; no factor of 2', 'Q': self.g.tolist(), 'offset': self.offset, 'variables': [{'index': yr, 'name': zp, 'qubit': yr, 'canonical_schedule_position': yr} for yr, zp in enumerate(x)], 'auxiliary_variables': [], 'decision_encoding': '1=NS, 0=EW', 'ising': {'convention': 'z_i=1-2*x_i; offset + sum h_i*z_i + sum_{i<j} J_ij*z_i*z_j', 'offset': self.yw, 'h': self.h.tolist(), 'J': self.d.tolist()}, 'circuit_energy_scale': self.scale, 'derivation': 'MODEL.md; conditional expectations computed by traffic-state recursion', 'conditional_expectations': 129}

    @classmethod
    def from_dict(cls, aw):
        return cls(float(aw['offset']), np.array(aw['Q'], dtype=float), float(aw['ising']['offset']), np.array(aw['ising']['h'], dtype=float), np.array(aw['ising']['J'], dtype=float))

def build_qubo(scenario):
    zn = TrafficModel(scenario)
    xo = zn.expected_cost()
    h = np.array([(zn.expected_cost({yr: 0}) - zn.expected_cost({yr: 1})) / 2 for yr in range(8)])
    d = np.zeros((8, 8))
    for yr, zx in combinations(range(8), 2):
        d[yr, zx] = sum(((1 - 2 * m) * (1 - 2 * xz) * zn.expected_cost({yr: m, zx: xz}) for m in (0, 1) for xz in (0, 1))) / 4
    g = 4 * d
    np.fill_diagonal(g, -2 * h - 2 * (d.sum(axis=0) + d.sum(axis=1)))
    offset = float(xo + h.sum() + d.sum())
    return Qubo(offset, g, xo, h, d)

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
c = np.arange(256)
b = tuple((c ^ 1 << al for al in range(8)))

def unpack(params, mixer='local'):
    if mixer not in ('global', 'local'):
        raise ValueError("Mixer must be 'global' or 'local'")
    params = np.asarray(params, dtype=float)
    cf = 1 if mixer == 'global' else 8
    if params.ndim != 1 or not len(params) or len(params) % (1 + cf):
        raise ValueError('Parameter vector has the wrong size for the mixer')
    ab = len(params) // (1 + cf)
    if not 1 <= ab <= 4 or not np.isfinite(params).all():
        raise ValueError('Use finite parameters and depth between 1 and 4')
    return (params[:ab], params[ab:].reshape(ab, cf))

def apply_mixer(bo, xd):
    bo = bo.copy()
    xd = np.broadcast_to(np.asarray(xd).reshape(-1), (8,))
    for al, v in enumerate(xd):
        xi, bg = (np.cos(v), -1j * np.sin(v))
        xf = bo.reshape(-1, 2, 1 << al)
        m, xz = (xf[:, 0, :].copy(), xf[:, 1, :].copy())
        xf[:, 0, :] = xi * m + bg * xz
        xf[:, 1, :] = bg * m + xi * xz
    return bo

def statevector(aq, params, mixer='local'):
    ym, xd = unpack(params, mixer)
    yd = aq.centered_energies() / aq.scale
    bo = np.ones(256, dtype=complex) / 16
    for yl, xz in zip(ym, xd):
        bo = apply_mixer(bo * np.exp(-1j * yl * yd), xz)
    return bo

def value_and_gradient(yd, params, mixer='local'):
    ym, xd = unpack(params, mixer)
    bo = np.ones(256, dtype=complex) / 16
    p, q = ([], [])
    for yl, xz in zip(ym, xd):
        bo = bo * np.exp(-1j * yl * yd)
        p.append(bo)
        bo = apply_mixer(bo, xz)
        q.append(bo)
    value = float(np.abs(bo) ** 2 @ yd)
    o = yd * bo
    yz, xu = (np.zeros_like(ym), np.zeros_like(xd))
    for zy in range(len(ym) - 1, -1, -1):
        zf = np.array([2 * np.imag(np.vdot(o, q[zy][yi])) for yi in b])
        xu[zy] = zf.sum() if mixer == 'global' else zf
        o = apply_mixer(o, -xd[zy])
        yz[zy] = 2 * np.imag(np.vdot(o, yd * p[zy]))
        o = o * np.exp(1j * ym[zy] * yd)
    return (value, np.r_[yz, xu.ravel()])

def circuit(aq, params, zj=True, mixer='local'):
    ym, xd = unpack(params, mixer)
    an = QuantumCircuit(8, 8) if zj else QuantumCircuit(8)
    an.h(range(8))
    for yl, xz in zip(ym, xd):
        for yr, h in enumerate(aq.h):
            if h:
                an.rz(float(2 * yl * h / aq.scale), yr)
        for yr in range(8):
            for zx in range(yr + 1, 8):
                if aq.d[yr, zx]:
                    an.rzz(float(2 * yl * aq.d[yr, zx] / aq.scale), yr, zx)
        for al, v in enumerate(np.broadcast_to(xz, (8,))):
            an.rx(float(2 * v), al)
    if zj:
        an.measure(range(8), range(8))
    return an

@dataclass
class TrainingResult:
    params: np.ndarray
    evaluations: int
    history: list
    zr: float
    yo: int
    optimizer: str
    mixer: str

class EvaluationBudgetReached(Exception):
    pass

def train(aq, bk, ab=4, zh=300, mixer='local', optimizer='adjoint'):
    if not 1 <= ab <= 4:
        raise ValueError('QAOA depth must be between 1 and 4')
    if not 2 * ab + 2 <= zh <= 300:
        raise ValueError('Evaluation budget must be between 2*p+2 and 300')
    if mixer not in ('global', 'local') or optimizer not in ('cobyla', 'adjoint'):
        raise ValueError('Unknown mixer or optimizer')
    bd = np.random.default_rng(bk)
    yd = aq.centered_energies() / aq.scale
    history = []
    xb, xc = (None, float('inf'))
    yp = 0
    cf = 1 if mixer == 'global' else 8
    ba = 3 if optimizer == 'cobyla' or mixer == 'global' else 2
    bb = min(ba, zh // (2 * ab + 2))
    xh = [zh // bb + int(at < zh % bb) for at in range(bb)]
    for bz, xg in enumerate(xh):
        if bz == 0:
            yu = np.r_[np.linspace(0.35, 1.0, ab), np.repeat(np.linspace(-0.65, -0.2, ab), cf)]
        elif mixer == 'global':
            yu = np.r_[bd.uniform(0.05, 2.5, ab), bd.uniform(-1.3, -0.05, ab)]
        else:
            yu = np.r_[bd.uniform(0.1, 2.5, ab), np.repeat(bd.uniform(-1.2, -0.1, ab), cf) + bd.normal(0, 0.15, ab * cf)]
        ze = 0

        def objective(params):
            nonlocal xb, xc, yp, ze
            if ze >= xg or len(history) >= zh:
                raise EvaluationBudgetReached()
            if optimizer == 'adjoint':
                value, yn = value_and_gradient(yd, params, mixer)
                yp += 1
            else:
                ak = statevector(aq, params, mixer)
                value = float(np.abs(ak) ** 2 @ yd)
            ze += 1
            bf = {'evaluation': len(history) + 1, 'restart': bz, 'normalized_expectation': value, 'params': np.asarray(params).tolist()}
            if optimizer == 'adjoint':
                bf['gradient_norm'] = float(np.linalg.norm(yn))
            history.append(bf)
            if value < xc:
                xb, xc = (np.array(params, copy=True), value)
            return (value, yn) if optimizer == 'adjoint' else value
        try:
            if optimizer == 'cobyla':
                minimize(objective, yu, method='COBYLA', options={'maxiter': xg, 'rhobeg': 0.6, 'tol': 1e-05})
            else:
                minimize(objective, yu, jac=True, method='L-BFGS-B', options={'maxiter': xg, 'maxfun': xg, 'ftol': 1e-12, 'gtol': 1e-08})
        except EvaluationBudgetReached:
            pass
    return TrainingResult(xb, len(history), history, xc, yp, optimizer, mixer)

def sample_final(aq, params, bk, mixer='local'):
    bj = Statevector.from_instruction(circuit(aq, params, zj=False, mixer=mixer))
    by = Statevector(statevector(aq, params, mixer))
    if not bj.equiv(by, rtol=1e-10, atol=1e-10):
        raise AssertionError('Fast training simulator disagrees with the Qiskit circuit')
    bj.seed(int(bk))
    xr = {str(zz): int(zo) for zz, zo in bj.sample_counts(4096).items()}
    return (dict(sorted(xr.items())), bj.probabilities())

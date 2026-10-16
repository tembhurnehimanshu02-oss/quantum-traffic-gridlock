import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
i = Path(__file__).resolve().parent
sys.path.insert(0, str(i / 'model'))
from constructions import e, build, describe
from scorer import validate_scenario
from qaoa import train, sample_final, circuit
from evaluation import assess
from qiskit import qasm3

def main():
    ac = argparse.ArgumentParser(description=__doc__)
    ac.add_argument('--scenario', required=True)
    ac.add_argument('--seed', required=True, type=int)
    ac.add_argument('--out', required=True)
    ac.add_argument('--method', choices=e, default='boltzmann_015')
    w = ac.parse_args()
    if w.seed < 0:
        ac.error('Seed must be nonnegative')
    bg = json.loads(Path(w.scenario).read_text())
    validate_scenario(bg)
    ax = Path(w.out)
    ax.mkdir(parents=True, exist_ok=True)
    if (ax / 'scenario.json').resolve() == Path(w.scenario).resolve():
        ac.error('Use an output directory separate from the input scenario')
    bn = time.perf_counter()
    al = build(bg, w.method)
    bl = [int(xi.generate_state(1)[0]) for xi in np.random.SeedSequence(w.seed).spawn(2)]
    cz = train(al, bl[0])
    xr, ai = sample_final(al, cz.params, bl[1])
    zl, xv = assess(bg, al, xr, ai)
    yg = {'scenario': bg, 'qubo': describe(al, w.method), 'counts': xr, 'decoded_counts': xv, 'metrics': zl, 'schedule': {'bits': zl['qaoa']['best_observed']['bits'], 'canonical_order': 'A1 A2 A3 A4 B1 B2 B3 B4', 'selection': 'Lowest official cost among raw samples, canonical tie break; no repair'}, 'training': {'seed': w.seed, 'training_seed': bl[0], 'sampling_seed': bl[1], 'params': cz.params.tolist(), 'evaluations': cz.evaluations, 'gradients': cz.yo, 'history': cz.history, 'qubits': 8, 'depth': 4, 'mixer': 'local', 'optimizer': 'adjoint', 'shots': 4096, 'normalized_expectation': cz.zr, 'seconds': time.perf_counter() - bn}}
    for zp, xt in yg.items():
        (ax / f'{zp}.json').write_text(json.dumps(xt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (ax / 'circuit.qasm').write_text(qasm3.dumps(circuit(al, cz.params)), encoding='utf-8')
    print(json.dumps({'method': w.method, 'mean_cost': zl['qaoa']['mean_official_cost'], 'lift': zl['qaoa']['lift'], 'seconds': time.perf_counter() - bn}, indent=2))
if __name__ == '__main__':
    main()

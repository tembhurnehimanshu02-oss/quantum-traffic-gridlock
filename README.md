# Quantum Traffic Gridlock

A traffic signal optimizer using a Boltzmann weighted QUBO and the Quantum
Approximate Optimization Algorithm (QAOA). This repository contains the final
uncommented penultimate solver with simplified lowercase variable names.

## Contents

- `run.py`: command-line entry point.
- `model/constructions.py`: traffic-derived QUBO fitting rules.
- `model/traffic.py`: queue dynamics and classical greedy control.
- `model/qubo.py`: QUBO and Ising representation.
- `model/qaoa.py`: circuit construction, training and raw sampling.
- `model/scorer.py`: official scoring logic with style edits.
- `model/evaluation.py`: baselines and sampling metrics.
- `requirements.txt`: pinned dependencies.

## Setup

Use Python 3.11. Create and activate a virtual environment, then install:

```text
python -m pip install -r requirements.txt
```

Dependencies are NumPy 2.4.6, SciPy 1.17.1 and Qiskit 2.5.2, with OpenQASM 3
import support. Install before offline execution; no network is used by the solver.

## Run

Provide a scenario JSON file using the challenge's input schema:

```text
python run.py --scenario scenario.json --seed 0 --out results/run0
```

Use an output directory separate from the input scenario. Scenario files and
saved experiment results are supplied separately and are not included here.
The same command accepts R, H or a new compatible scenario without code changes.
Optional `--method uniform` selects the uniform QUBO comparison rule.

The input contains arrivals for A and B, each with NS and EW arrays of four
nonnegative integers, service_capacity, storage_capacity, overflow_weight,
switch_weight, and a link object with from=A, to=B, phase=NS or EW, and fraction.

## Selected method

The model has two intersections and four slots, with eight decisions ordered
A1 A2 A3 A4 B1 B2 B3 B4. A bit of 1 gives NS green; 0 gives EW green.

For each scenario the builder evaluates all 256 traffic histories with its own
queue dynamics. It fits a quadratic function to the resulting costs using
weighted least squares. The default boltzmann_015 rule gives greater influence
to low-cost histories with a normalized temperature of 0.15 and a 2% uniform
component. It does not find an optimal bitstring and encode that answer.

For z=1-2x, the feature vector contains 1, the eight z values and their 28 pairwise
products. The coefficient vector minimizes sum_x w(x) [C(x)-F(x) theta]^2.
Weights are w=0.02+0.98*b/mean(b), with
b=exp(-(C-mean(C))/(0.15*max(std(C),1))). A common exponential rescaling makes
this computation numerically stable. The resulting Ising fields and couplings
define the circuit's cost operator; its constant term is omitted as global phase.
Some generated metadata retains historical references to METHODS.md or MODEL.md;
the selected construction and model are documented in this README.

The circuit uses eight qubits, four QAOA layers and 36 trainable angles: four
cost angles and 32 independent X mixer angles. Adjoint gradients train the
angles with L-BFGS-B, using two starts and at most 300 objective evaluations.
The final circuit supplies exactly 4096 raw samples per seed. No sample repair,
postselection or classical replacement of the sampled distribution is applied.

## Outputs

- `qubo.json`: matrix, offset, Ising coefficients and variable mapping.
- `circuit.qasm`: bound OpenQASM 3 circuit.
- `counts.json`: 4096 raw counts in Qiskit bit order.
- `schedule.json`: best officially scored schedule observed in the samples.
- `decoded_counts.json`: counts in canonical traffic bit order.
- `training.json`: trained angles, objective trace, evaluation count and seeds.
- `metrics.json`: official scores, baselines, lift and elite enrichment.
- `scenario.json`: copy of the input.

## Results and limitations

Five-seed raw mean costs are 75.484375 on R and 235.748974609375 on H. Their
exact optima are 72 and 234. The style-edited source reproduced the frozen
solver's counts, parameters, metrics and exported circuits across all ten runs.

Classical enumeration establishes exact benchmark optima. Training minimizes
expected QUBO cost, without receiving an optimal schedule. Construction already
examines all 256 traffic histories; exact classical optimization is easy at this
size. This work does not demonstrate quantum advantage, large-network scalability
or noisy-hardware performance. Experiments use ideal CPU statevector simulation.
The scorer logic was checked against the unchanged organizer scorer, but its
packaged source has comment and variable-name edits.

## Submission documents

- [Technical report](QTG_Report.pdf)
- [AI-use disclosure](AI_Usage_Disclosure.pdf)

Saved experiment outputs are available in the separate submission package.

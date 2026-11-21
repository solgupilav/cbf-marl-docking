# cbf-marl-docking

Code for the paper "Safe Multi-Agent Spacecraft Docking with Control Barrier
Functions and Reinforcement Learning" by Selim Olgu Pilav, accepted to the 2027
IEEE Aerospace Conference.

[Paper (PDF)](https://solgupilav.github.io/papers/pilav_2027_safe_multi_agent_docking.pdf)

Nine 6U CubeSat-sized chasers dock at the same time on ports of a target
spacecraft. The dynamics are Clohessy–Wiltshire translation and quaternion
attitude. All chasers share one policy, trained with multi-agent PPO (MAPPO).
Each chaser runs the policy on its own observation: its own state and the two
nearest chasers, through a noisy sensor model. A safety filter, a quadratic program built on high-order control
barrier functions, changes each command as little as possible. It keeps the
chasers apart, limits their approach speed and keeps them out of the target.
The filter runs in training and in deployment.

## Main result

5 training seeds × 1000 Monte Carlo episodes, 4 cm / 2° capture gate:

| Metric | Value |
|---|---|
| All nine chasers docked | **99.54%** (95% CI 99.31–99.69%) |
| Target contacts | 0 of 5000 episodes |
| Inter-agent contacts | 0 of 5000 episodes |
| Linear Δv per chaser | 0.345 m/s |

[`results/PAPER_NUMBERS.txt`](results/PAPER_NUMBERS.txt) gives all the results
in the paper. `scripts/paper_numbers.py` makes this file from the stored
results.

## Contents

```
src/marl_docking_6dof/
  envs/docking_6dof_env.py   docking environment (TorchRL)
  safety/cbf.py              HOCBF safety filter
  tasks.py                   BenchMARL task and default configuration
scripts/                     training, evaluation, analysis and figures
reproduce/                   one script for each step
results/                     stored results for all tables and figures
checkpoints/                 the six trained policies of the paper
tests/                       tests for the filter and the environment
```

## Installation

All results use Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests
```

## Reproduce the paper

Draw the figures from the stored results:

```bash
bash reproduce/04_figures.sh
```

Figures 3 and 6 also need the trained policies. The script makes their data
first.

Evaluate the five policies (Table 10). Each seed prints `MATCH` if the result
is the same as the stored result. This takes a few minutes on a laptop CPU.

```bash
bash reproduce/02_evaluate_main.sh
```

Run the other evaluations (Tables 11-17), all or by block:

```bash
bash reproduce/03_evaluate_ablations.sh
bash reproduce/03_evaluate_ablations.sh filter lqr
python scripts/collate_sensitivity.py
```

Train the policies again (25M frames each, several hours on a CPU):

```bash
bash reproduce/01_train.sh
bash reproduce/01_train.sh ablations
```

After each run, `scripts/select_checkpoint.py` selects the checkpoint to
report, as in the paper: the best mean success over three neighbouring
checkpoints on a screening seed.

Set `PY` to use a different Python, for example
`PY=.venv/bin/python bash reproduce/02_evaluate_main.sh`.

## Results and their sources

| Paper | How to reproduce | Stored result |
|---|---|---|
| Figs. 1-7 | `04_figures.sh` | `results/figure_data/`, `results/main/` |
| Table 10, Figs. 4, 5, 7 | `02_evaluate_main.sh` | `results/main/` |
| Table 11: filter ablation | `03_evaluate_ablations.sh filter` | `results/ablations/` |
| Table 12: LQR with the filter | `03_evaluate_ablations.sh lqr` | `results/lqr/lqr_*.json` |
| Table 13: fuel against LQR | `03_evaluate_ablations.sh lqr` | `results/lqr/fuel_reference.txt` |
| Table 14: keep-out geometry | `03_evaluate_ablations.sh keepout` | `results/keepout/` |
| Table 15: actuator uncertainty | `03_evaluate_ablations.sh actuator` | `results/sensitivity/actuator_*` |
| Table 16: swarm size | `03_evaluate_ablations.sh scaling` | `results/sensitivity/scale_*` |
| Table 17: compute cost | `03_evaluate_ablations.sh compute` | `results/compute/flop_count.txt` |
| Filter paths per step (Computational Cost) | `scripts/filter_path_stats.py` | `results/compute/filter_path_stats.json` |
| int8 and fp16 weights | `03_evaluate_ablations.sh quant` | `results/quantisation/` |
| Probabilistic safety (Discussion) | `scripts/prob_cbf_exposure2.py`, then `scripts/prob_cbf_paper_numbers.py` | `results/probabilistic_safety/` |

## How this code was developed

This code was written with AI coding assistance by Claude, as disclosed in the
paper's acknowledgements. The author developed the study from the research
question and methods to the experiment design and interpretation of the
results. The author reviewed all code used in the study, including the results
produced by the code. Every reported number was checked against the stored
evaluation outputs in `results/`.

## License

MIT. See [LICENSE](LICENSE).

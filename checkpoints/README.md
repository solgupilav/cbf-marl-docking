# Trained policies

The six policies of the paper. Each is a BenchMARL run folder:

```
checkpoints/<name>/
  config.pkl               training configuration
  checkpoints/policy.pt    trained weights
```

| Folder | Policy | Used for |
|---|---|---|
| `seed0` ... `seed4` | Training seeds 0-4 | Tables 10 and 12-17 |
| `ablation_trained_without_filter` | Trained without the safety filter (seed 0) | Table 11 |

The files are the training checkpoints with the rollout buffer cut to one
entry. Evaluation does not use the buffer, and each file is about 1 MB instead
of 61 MB. All evaluation results are the same as the stored results.

`scripts/mc_eval_6dof.py` loads each policy from a temporary copy, so these
folders do not change when you evaluate.

To train the policies again, see `reproduce/01_train.sh`.

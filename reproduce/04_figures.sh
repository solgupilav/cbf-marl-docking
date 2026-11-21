#!/usr/bin/env bash
# Draw the paper figures into figures/.
#
#   bash reproduce/04_figures.sh
#
# The figures use the stored results in results/. Two inputs are not stored,
# because they are large. This script makes them from the checkpoints:
#   Fig. 3  results/figure_data/figdata.npz, a rollout of seed 3 in its
#           training setup.
#   Fig. 6  results/figure_data/intervention_seed*.npz, the per-step filter
#           intervention.
source "$(dirname "$0")/common.sh"

if [ -f "$CKPT_DIR/seed3/checkpoints/policy.pt" ]; then
  if [ ! -f results/figure_data/figdata.npz ]; then
    echo "=== rollout for Fig. 3"
    "$PY" scripts/make_figure_data.py --checkpoint "$CKPT_DIR/seed3/checkpoints/policy.pt" \
      --episodes 400 --out-dir results/figure_data > logs/figdata.log 2>&1 || echo "    FAILED"
  fi
  for i in 0 1 2 3 4; do
    if [ ! -f "results/figure_data/intervention_seed$i.npz" ] && [ -f "$CKPT_DIR/seed$i/checkpoints/policy.pt" ]; then
      echo "=== intervention arrays, seed $i"
      "$PY" scripts/mc_eval_6dof.py --checkpoint "$CKPT_DIR/seed$i/checkpoints/policy.pt" \
        --episodes 1000 --seed "$EVAL_SEED" "${REPORT[@]}" \
        --json-out "$OUT/intervention_seed$i.json" --arrays-out "results/figure_data/intervention_seed$i.npz" \
        > "logs/intervention_seed$i.log" 2>&1 || echo "    FAILED"
    fi
  done
fi

FIGDIR=${FIGDIR:-figures} "$PY" scripts/make_paper_figures.py

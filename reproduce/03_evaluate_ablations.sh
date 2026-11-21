#!/usr/bin/env bash
# Evaluations behind Tables 11-17 of the paper. Give block names to run only
# those blocks:
#
#   bash reproduce/03_evaluate_ablations.sh filter lqr
#
# Blocks: filter lqr keepout actuator scaling quant compute
# With no arguments, all blocks run (about 60 evaluations).
# Results go to outputs/. Each output has the same name as its stored result
# in results/. The paper values are in results/PAPER_NUMBERS.txt.
source "$(dirname "$0")/common.sh"

BLOCKS=("$@")
[ ${#BLOCKS[@]} -eq 0 ] && BLOCKS=(filter lqr keepout actuator scaling quant compute)
want() { [[ " ${BLOCKS[*]} " == *" $1 "* ]]; }

# Table 11: seed 3 with the filter on and off, and the policy trained without
# the filter.
if want filter; then
  evaluate seed3 filter_on
  evaluate seed3 filter_off --cbf off
  evaluate ablation_trained_without_filter trained_without_filter_eval_on --cbf on
  evaluate ablation_trained_without_filter trained_without_filter_eval_off --cbf off
fi

# Table 12: LQR nominal without and with the filter.
# Table 13: LQR fuel reference without collision avoidance.
if want lqr; then
  evaluate seed3 lqr_no_filter --nominal lqr --cbf off
  evaluate seed3 lqr_with_filter --nominal lqr --cbf on
  # The checkpoint gives only the task configuration.
  "$PY" scripts/lqr_baseline_6dof.py --checkpoint "$(ckpt seed3)" --episodes 400 \
    --goal-tolerance 0.04 --attitude-gate 0.0349 \
    > "$OUT/fuel_reference.txt" 2> logs/fuel_reference.log || echo "    LQR reference FAILED"
fi

# Table 14: keep-out geometry, 5 seeds, 300 and 600 steps.
if want keepout; then
  for i in 0 1 2 3 4; do
    for H in 300 600; do
      sfx=$([ $H = 300 ] && echo _h300)
      evaluate "seed$i" "seed${i}_m8$sfx" --max-steps $H \
        --keepout-mode superellipsoid --target-lambda 0.15
      evaluate "seed$i" "seed${i}_m16$sfx" --max-steps $H \
        --keepout-mode superellipsoid --keepout-cover-box on --target-lambda 0.15
      evaluate "seed$i" "seed${i}_halfspace$sfx" --max-steps $H
    done
  done
fi

# Table 15: actuator uncertainty, 5 seeds.
if want actuator; then
  for i in 0 1 2 3 4; do
    evaluate "seed$i" "actuator_typical_seed$i" --actuator-uncertainty typical
    evaluate "seed$i" "actuator_high_seed$i" --actuator-uncertainty high
  done
fi

# Table 16: swarm size, 5 seeds.
if want scaling; then
  for i in 0 1 2 3 4; do
    for N in 4 6 12 16 20; do
      evaluate "seed$i" "scale_n${N}_seed$i" --n-agents "$N"
    done
  done
fi

# Seed-3 policy with fp16 and int8 weights (Computational Cost section).
if want quant; then
  evaluate seed3 seed3_fp16 --quantize-weights fp16
  evaluate seed3 seed3_int8 --quantize-weights int8
fi

# Table 17: FLOPs per control step. Stage 1 saves the state, stage 2 counts.
if want compute; then
  "$PY" scripts/count_flops.py --dump > logs/flop_count.log 2>&1 \
    && "$PY" scripts/count_flops.py > "$OUT/flop_count.txt" 2>> logs/flop_count.log \
    || echo "    FLOP count FAILED, see logs/flop_count.log"
fi

echo "Done. Pool the 5-seed cells with: python scripts/collate_sensitivity.py"

#!/usr/bin/env bash
# Train the policies reported in the paper.
#
#   bash reproduce/01_train.sh            # the five baseline seeds
#   bash reproduce/01_train.sh ablations  # also the policy trained without the filter
#
# Each run is 25M frames (several hours on a laptop CPU). The output goes to
# runs/<name>/, with the same layout as checkpoints/<name>/.
# After training, scripts/select_checkpoint.py selects the checkpoint to
# report: the best three-checkpoint mean success on the screening seed (1234),
# then a 1000-episode check on the evaluation seed (24680). The result is in
# outputs/select/<name>_selected.json.
source "$(dirname "$0")/common.sh"

# Baseline: sensor noise on the observation, delta-v cost (angular weight 32),
# safety features 0 and 2, and a curriculum from 15 cm / 10 deg to
# 4 cm / 2 deg. The filter is on, with the superellipsoid (m = 8) keep-out.
BASE=(--sensor-noise --fuel --fuel-ang-scale 32 --safety-obs-features 0,2
      --attitude-curriculum --end-attitude 0.0349 --end-tolerance 0.04
      --max-frames 25000000 --checkpoint-every 5 --no-save-eval-videos)

train() {  # train <run-name> <seed> [extra flags...]
  local name=$1 seed=$2; shift 2
  echo "=== training $name (seed $seed)"
  "$PY" -u scripts/train_benchmarl_6dof.py "${BASE[@]}" --seed "$seed" "$@" \
    --save-folder "runs/$name" > "logs/train_$name.log" 2>&1 \
    || echo "    FAILED, see logs/train_$name.log"
}

pick() {  # pick <run-name> [extra mc_eval flags...]
  local name=$1; shift
  local mc=()
  for a in "${REPORT[@]}" "$@"; do mc+=("--mc-arg=$a"); done
  echo "=== selecting the checkpoint of $name"
  "$PY" scripts/select_checkpoint.py --run-dir "runs/$name" --label "$name" \
    --refine-top 3 --final-seed "$EVAL_SEED" "${mc[@]}" \
    > "logs/select_$name.log" 2>&1 || echo "    FAILED, see logs/select_$name.log"
}

for s in 0 1 2 3 4; do
  mkdir -p "runs/seed$s"
  train "seed$s" "$s"
  pick "seed$s"
done

if [ "${1:-}" = "ablations" ]; then
  mkdir -p runs/ablation_trained_without_filter
  # Trained without the safety filter (Table 11). This run used seed 0 and a
  # checkpoint every 20 batches.
  train ablation_trained_without_filter 0 --no-cbf --checkpoint-every 20
  pick ablation_trained_without_filter --cbf on
fi

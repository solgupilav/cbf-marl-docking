# Shared settings for the reproduction scripts. Source this file.
#
#   PY        Python interpreter (default: python)
#   CKPT_DIR  folder holding the trained policies (default: checkpoints)
#   OUT       folder for new evaluation results (default: outputs)

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PY=${PY:-python}
CKPT_DIR=${CKPT_DIR:-checkpoints}
OUT=${OUT:-outputs}
export OUT
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT" logs

# Scoring configuration of all reported numbers: 4 cm / 2 deg capture gate,
# half-space keep-out with pole 0.04, and the filter on the sensed state.
REPORT=(--goal-tolerance 0.04 --attitude-gate 0.0349 --keepout-mode halfspace
        --target-lambda 0.04 --cbf-sensed-state on)

# Evaluation seed (not used in training).
EVAL_SEED=24680
EPISODES=${EPISODES:-1000}

ckpt() {  # ckpt <name>  ->  checkpoints/<name>/checkpoints/policy.pt, or exit
  local p="$CKPT_DIR/$1/checkpoints/policy.pt"
  [ -f "$p" ] || { echo "Missing checkpoint: $p (see checkpoints/README.md)" >&2; exit 1; }
  echo "$p"
}

evaluate() {  # evaluate <checkpoint-name> <output-name> [extra mc_eval flags...]
  local ck name
  ck=$(ckpt "$1") || exit 1; name=$2; shift 2
  echo "=== $name"
  "$PY" scripts/mc_eval_6dof.py --checkpoint "$ck" \
    --episodes "$EPISODES" --seed "$EVAL_SEED" "${REPORT[@]}" "$@" \
    --json-out "$OUT/$name.json" --arrays-out "$OUT/$name.npz" \
    > "logs/$name.log" 2>&1 || echo "    FAILED, see logs/$name.log"
}

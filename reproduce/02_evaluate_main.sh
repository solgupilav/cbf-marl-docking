#!/usr/bin/env bash
# Main result: 5 seeds x 1000 Monte Carlo episodes.
#
#   bash reproduce/02_evaluate_main.sh
#
# Writes outputs/seed<i>.json and .npz and compares each with
# results/main/seed<i>.json. The evaluation is deterministic, so each
# seed must print MATCH.
source "$(dirname "$0")/common.sh"

for i in 0 1 2 3 4; do
  evaluate "seed$i" "seed$i"
done

"$PY" - <<'EOF'
import json, os
keys = ["complete_success_rate", "any_target_collision_rate",
        "any_interagent_collision_rate", "mean_episode_len"]
out = os.environ.get("OUT", "outputs")
rows = []
for i in range(5):
    new = os.path.join(out, f"seed{i}.json")
    if not os.path.exists(new):
        print(f"seed {i}: no output"); continue
    a = json.load(open(f"results/main/seed{i}.json")); b = json.load(open(new))
    ok = all(abs(a[k] - b[k]) < 1e-6 for k in keys)
    rows.append(b)
    print(f"seed {i}: paper {a['complete_success_rate']:.4f}  re-run "
          f"{b['complete_success_rate']:.4f}  {'MATCH' if ok else 'MISMATCH'}")
if rows:
    n = sum(r["episodes"] for r in rows)
    succ = sum(r["complete_success_rate"] * r["episodes"] for r in rows) / n
    tgt = sum(round(r["any_target_collision_rate"] * r["episodes"]) for r in rows)
    inter = sum(round(r["any_interagent_collision_rate"] * r["episodes"]) for r in rows)
    print(f"pooled over {n} episodes: success {100*succ:.2f}%  "
          f"target contacts {tgt}  inter-agent contacts {inter}")
EOF

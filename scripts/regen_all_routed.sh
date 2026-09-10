#!/bin/bash
# Regenerate all routed JSONs after the eager-1Q / layout-field fix.
# Same models / beam widths / seed as the original runs (recovered from *_route.log + train.md).
set -u
cd "$(dirname "$0")/../src" || exit 1
export PYTHONPATH=.

COMMON="--circuit-dir ../benchmark/nam_circs \
  --topo ../traindata/topo/tianyan176_20q.json \
  --label-map ../traindata/topo/tianyan176_20q_labels.json \
  --max-num-qubits 20 --no-fidelity --seed 0"

run() {  # run <dir> <model_path> <beam>
  local dir="$1" model="$2" beam="$3"
  local extra=""
  [ "$beam" != "0" ] && extra="--beam-width $beam"
  echo "=== [$(date +%H:%M:%S)] regen $dir (model=$model beam=$beam) ==="
  python3 -u -m routing.rl.generate_routing \
    --model "$model" --model-name "$dir" \
    --out-dir "../benchmark/routed/$dir" \
    --out-json "../benchmark/routed/${dir}_summary.json" \
    $COMMON $extra || echo "!!! FAILED: $dir"
}

run l05             ../models/policy_tianyan20q_laymix_l05_eta05.pt 0
run l05_beam3       ../models/policy_tianyan20q_laymix_l05_eta05.pt 3
run l05_beam5       ../models/policy_tianyan20q_laymix_l05_eta05.pt 5
run l05_nam         ../models/policy_l05_nam_phase1.pt              0
run ph2v4           ../models/policy_ph2_v4.pt                      0
run ph2v4_beam3     ../models/policy_ph2_v4.pt                      3
run ph2v4_beam5     ../models/policy_ph2_v4.pt                      5
run nam_l05_v2      ../models/policy_nam_l05_finetune_v2.pt         0
run nam_l05_v2_beam3 ../models/policy_nam_l05_finetune_v2.pt        3
run nam_p2a         ../models/policy_p2a_ema_best.pt                0
run nam_p2a_beam3   ../models/policy_p2a_ema_best.pt                3
run nam_p2b         ../models/policy_p2b_ema_best.pt                0
run nam_p2b_beam3   ../models/policy_p2b_ema_best.pt                3
run nam_p2c         ../models/policy_p2c_ema_best.pt                0
run nam_p2c_beam3   ../models/policy_p2c_ema_best.pt                3
run nam_sref_v1     ../models/policy_nam_sref_ft_v1_ema_best.pt     0
run nam_sref_v1_beam3 ../models/policy_nam_sref_ft_v1_ema_best.pt   3
run nam_sref_final  ../models/policy_nam_sref_ft_v1.pt              0
run nam_traj_v1     ../models/policy_nam_traj_ft_v1.pt              0
run nam_traj_v1_beam3 ../models/policy_nam_traj_ft_v1.pt            3

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="

#!/bin/bash
# Runs ON the box (80GB). Predictor-validation expansion: three new reward
# settings, each with (a) pre-flight coordinate 2 (cross-prompt gradient
# cosine), (b) one beta=0 FlowGRPO run (4h, seed 42), (c) the TFG dose-response
# {2,5,10,20} (coordinate 1, the slope) + RL arm on 150 prompts.
# With the five settings already measured this gives the 8-point predictor.
#
# Usage: HF_TOKEN=... bash scripts/cloud/run_rw8.sh [all|tfg|rl]
#   all : everything on one 80GB box (default)
#   tfg : eval-only half (pre-flight cosine, TFG dose-responses, ensemble dose) — fits a 40GB A100
#   rl  : the beta=0 GRPO trainings + base/RL eval only — needs 80GB
# Env:   RW8_KEYS="hpsv2 siglip clip_pickapic"  RW8_WALL=4  RW8_SEEDBASES=0
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
export HR_PROJECT_ROOT="$PROJECT_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?}"
export FG_GROUP=8 FG_NBPE=8
STAGE="${1:-all}"
KEYS="${RW8_KEYS:-hpsv2 siglip clip_pickapic}"
WALL="${RW8_WALL:-4}"
SEEDBASES="${RW8_SEEDBASES:-0}"
log "=== RW8 start: keys [$KEYS], ${WALL}h beta=0 seed 42 each, dose-response rho 2,5,10,20 ==="

# 0. environment + predictor load checks (fail fast, before any GPU-hours)
if [ "$STAGE" = "tfg" ]; then
  [ -f .remote_setup_done ] || { HF_TOKEN="$HF_TOKEN" bash scripts/cloud/remote_setup.sh && touch .remote_setup_done || { log "FATAL: remote_setup"; exit 1; }; }
  python3 -m pip install --quiet --no-deps open_clip_torch && python3 -m pip install --quiet ftfy regex torchvision
  mkdir -p repos; [ -d repos/flow_grpo ] || git clone --quiet --depth 1 https://github.com/yifan123/flow_grpo repos/flow_grpo
  python3 - <<'PY' || { log "FATAL: predictor check failed"; exit 1; }
import sys, torch; sys.path.insert(0, ".")
from scripts.hpsv2_predictor import HPSv2PromptPredictor
from scripts.siglip_predictor import SiglipPromptPredictor
x = torch.rand(1, 3, 512, 512, device="cuda") * 2 - 1
for P in (HPSv2PromptPredictor, SiglipPromptPredictor):
    p = P(device="cuda"); s = p(x, ["a photo of a cat"]); assert torch.isfinite(s).all(); print("  ok", P.__name__, s.tolist(), flush=True); del p; torch.cuda.empty_cache()
PY
else
  for KEY in $KEYS; do
    HF_TOKEN="$HF_TOKEN" FG_REWARD=$KEY FG_SETUP_ONLY=1 bash scripts/cloud/run_flowgrpo_extra.sh \
      || { log "FATAL: setup/predictor check failed for $KEY"; exit 1; }
  done
fi

# 1. eval prompt files
PUB_PROMPTS="$PROJECT_ROOT/outputs/eval_prompts_pub.txt"
python3 - scripts/cloud/prompts_smoke.json scripts/cloud/prompts_train.json "$PUB_PROMPTS" <<'PY'
import json, sys, pathlib
smoke = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[1]).read_text())]
train = [d["prompt"] for d in json.loads(pathlib.Path(sys.argv[2]).read_text())]
pathlib.Path(sys.argv[3]).write_text("\n".join(smoke + [p for p in train if p not in set(smoke)][:120]) + "\n")
PY
PICK_PROMPTS="$PROJECT_ROOT/repos/flow_grpo/dataset/pickscore/test.txt"

# 2. pre-flight coordinate 2
SETTINGS=""
for KEY in $KEYS; do
  case "$KEY" in hpsv2) SETTINGS="$SETTINGS,hpsv2:geneval";; siglip) SETTINGS="$SETTINGS,siglip:geneval";; clip_pickapic) SETTINGS="$SETTINGS,clip:pickapic";; esac
done
if [ "$STAGE" != "rl" ]; then
  python3 scripts/cloud/pdep_measure.py --settings "${SETTINGS#,}" --out outputs/study_pdep_rw8.json \
    || log "WARN: pdep failed (continuing)"
fi

# 3. per setting: train + eval
lora_of() { ls -d "$PROJECT_ROOT"/outputs/checkpoints/$1/checkpoint-*/lora 2>/dev/null | sort -t- -k2 -V | tail -1; }
for KEY in $KEYS; do
  DEST="flowgrpo_${KEY}_b0_s42"
  if [ "$STAGE" != "tfg" ] && [ -z "$(lora_of $DEST)" ]; then
    log ">>> training $KEY beta=0 seed 42 (${WALL}h)"
    rm -rf "$PROJECT_ROOT/repos/flow_grpo/logs/$KEY" "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_$KEY"
    HF_TOKEN="$HF_TOKEN" FG_REWARD=$KEY FG_SEED=42 FG_BETA=0 bash scripts/cloud/run_flowgrpo_extra.sh "$WALL" \
      || log "WARN: $KEY train nonzero"
    [ -d "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_$KEY" ] && mv "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_$KEY" "$PROJECT_ROOT/outputs/checkpoints/$DEST"
  fi
  L=$(lora_of $DEST); RL_ARGS=()
  [ "$STAGE" = "tfg" ] && L=""
  if [ -n "$L" ]; then
    python3 scripts/cloud/verify_lora_load.py "$L" 2>&1 | tail -2 || log "WARN: verify_lora_load failed"
    RL_ARGS+=(--rl-lora "rl_b0_s42=$L")
  else
    log "WARN: no RL checkpoint for $KEY — dose-response only"
  fi
  case "$KEY" in
    hpsv2)         EV_REWARD=hpsv2;  EV_PROMPTS="$PUB_PROMPTS" ;;
    siglip)        EV_REWARD=siglip; EV_PROMPTS="$PUB_PROMPTS" ;;
    clip_pickapic) EV_REWARD=clip;   EV_PROMPTS="$PICK_PROMPTS" ;;
  esac
  case "$STAGE" in all) RHOS="2,5,10,20"; SUF="";; tfg) RHOS="2,5,10,20"; SUF="_tfg";; rl) RHOS=""; SUF="_rl";; esac
  log ">>> $KEY eval[$STAGE]: base + tfg{$RHOS} + ${#RL_ARGS[@]} RL args on 150 prompts x seeds[$SEEDBASES]"
  python3 scripts/cloud/eval_study_matrix.py \
    --reward "$EV_REWARD" --prompts-file "$EV_PROMPTS" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" \
    --tfg-rhos "$RHOS" --stack-rhos "" "${RL_ARGS[@]}" \
    --seed-bases "$SEEDBASES" \
    --out "outputs/study_rw8_${KEY}${SUF}_matrix.json" \
    || { log "FATAL: $KEY eval failed"; exit 1; }
done

# 3b. ensemble dose-response (eval-only, completes the 5th existing setting's slope)
if [ "$STAGE" != "rl" ] && [ ! -f outputs/study_rw8_ensemble_dose_matrix.json ]; then
  log ">>> ensemble dose-response: base + tfg{2,5,10} on 150 prompts x seeds[$SEEDBASES]"
  python3 scripts/cloud/eval_study_matrix.py \
    --reward ensemble --prompts-file "$PUB_PROMPTS" --num-prompts 150 --num-steps 40 \
    --ckpt-dir "$PROJECT_ROOT/outputs/checkpoints" --tfg-rhos 2,5,10 --stack-rhos "" \
    --seed-bases "$SEEDBASES" --out outputs/study_rw8_ensemble_dose_matrix.json \
    || log "WARN: ensemble dose-response failed"
fi
cp -f outputs/study_rw8_ensemble_dose_matrix.json outputs/checkpoints/ 2>/dev/null || true

# 4. bundle LoRAs + train logs (+ the ensemble dose matrix, since the pull is a fixed file list)
B=()
[ -f outputs/checkpoints/study_rw8_ensemble_dose_matrix.json ] && B+=("study_rw8_ensemble_dose_matrix.json")
for KEY in $KEYS; do L=$(lora_of flowgrpo_${KEY}_b0_s42); [ -n "$L" ] && B+=("${L#$PROJECT_ROOT/outputs/checkpoints/}"); done
[ ${#B[@]} -gt 0 ] && tar czf outputs/rw8_ckpts.tar.gz -C outputs/checkpoints "${B[@]}" || log "WARN: bundle failed"
log "=== RW8[$STAGE] done ==="

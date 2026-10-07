#!/bin/bash
# FlowGRPO (group=8, beta via FG_BETA) on an EXTRA reward setting, for the
# predictor-validation expansion (8 reward settings). Parametrised version of
# run_flowgrpo_aesthetic.sh; same upstream patches, same knobs (FG_GROUP,
# FG_NBPE, FG_SEED, FG_BETA, FG_MODEL).
#
#   FG_REWARD=hpsv2          HPSv2 (open_clip ViT-H preference model) on the 30 GenEval-style prompts
#   FG_REWARD=siglip         SigLIP so400m on the 30 GenEval-style prompts
#   FG_REWARD=clip_pickapic  CLIP-L on flow_grpo's Pick-a-Pic prompt set (prompt-distribution axis)
#
# Usage: HF_TOKEN=... FG_REWARD=hpsv2 bash scripts/cloud/run_flowgrpo_extra.sh [WALL_HOURS]
#        FG_SETUP_ONLY=1 -> install, patch, pre-cache, then exit before training.
set -euo pipefail
WALL_HOURS="${1:-4}"
WALL_SECONDS=$(python3 -c "print(int(float('$WALL_HOURS')*3600))")
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"
export HR_PROJECT_ROOT="$PROJECT_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
: "${HF_TOKEN:?HF_TOKEN required}"
KEY="${FG_REWARD:?FG_REWARD required (hpsv2|siglip|clip_pickapic)}"
case "$KEY" in
  hpsv2)         REWARD_FN='{"hpsv2": 1.0}';     DATASET_REL="dataset/ddrl_hpsv2" ;;
  siglip)        REWARD_FN='{"siglip": 1.0}';    DATASET_REL="dataset/ddrl_siglip" ;;
  clip_pickapic) REWARD_FN='{"clipscore": 1.0}'; DATASET_REL="dataset/pickscore" ;;
  *) log "unknown FG_REWARD $KEY"; exit 1 ;;
esac
log "=== FlowGRPO+$KEY (budget ${WALL_HOURS}h, group=${FG_GROUP:-2}, beta=${FG_BETA:-default}, seed=${FG_SEED:-42}) ==="
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

FLOW_GRPO_DIR="$PROJECT_ROOT/repos/flow_grpo"
mkdir -p "$PROJECT_ROOT/repos"
[ -d "$FLOW_GRPO_DIR" ] || { log "Cloning flow_grpo..."; git clone https://github.com/yifan123/flow_grpo "$FLOW_GRPO_DIR"; }
cd "$FLOW_GRPO_DIR"
log "flow_grpo at $(git rev-parse --short HEAD)"

log "Installing Python deps..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet --upgrade --force-reinstall pillow
python3 -m pip install --quiet 'diffusers>=0.31,<0.40' 'transformers==4.49.0' 'peft>=0.12' 'accelerate>=0.33' \
    'ml_collections' 'absl-py' 'einops' 'wandb' sentencepiece protobuf safetensors pyyaml tqdm huggingface_hub torchvision
sed -i.bak 's/bitsandbytes==[0-9.]*/bitsandbytes>=0.42.0/' setup.py pyproject.toml requirements.txt 2>/dev/null || true
sed -i.bak2 '/xformers/d' setup.py 2>/dev/null || true
python3 -m pip install --quiet -e .
if [ "$KEY" = "hpsv2" ]; then
  # open_clip without letting pip touch torch (the earlier pdep-ext run broke on a torch/dtype drift)
  python3 -m pip install --quiet --no-deps open_clip_torch && python3 -m pip install --quiet ftfy regex
fi
python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"

# --- Pre-cache SD3.5-M + the reward model, and prove the predictor loads ---
cd "$PROJECT_ROOT"
python3 - "$KEY" <<'PY'
import sys, time, torch
key = sys.argv[1]
from diffusers import StableDiffusion3Pipeline
t0 = time.time()
_ = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
print(f"  cached SD3.5-M in {time.time()-t0:.0f}s", flush=True)
sys.path.insert(0, ".")
if key == "hpsv2":
    from scripts.hpsv2_predictor import HPSv2PromptPredictor as P
elif key == "siglip":
    from scripts.siglip_predictor import SiglipPromptPredictor as P
else:
    from scripts.clip_predictor import CLIPPromptPredictor as P
pred = P(device="cuda")
x = torch.rand(2, 3, 512, 512, device="cuda") * 2 - 1
s = pred(x, ["a photo of a cat", "a photo of a dog"])
assert s.shape == (2,) and torch.isfinite(s).all(), s
print(f"  predictor {key} ok: {s.tolist()}", flush=True)
PY
cd "$FLOW_GRPO_DIR"

# --- Prompt set ---
if [ "$KEY" != "clip_pickapic" ]; then
  mkdir -p "$FLOW_GRPO_DIR/$DATASET_REL"
  python3 - "$PROJECT_ROOT/scripts/cloud/prompts_smoke.json" "$FLOW_GRPO_DIR/$DATASET_REL" <<'PROMPTS_PY'
import json, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
prompts = [d["prompt"] for d in json.loads(src.read_text())]
(out / "train.txt").write_text("\n".join(prompts) + "\n")
(out / "test.txt").write_text("\n".join(prompts) + "\n")
print(f"wrote {len(prompts)} prompts -> {out}")
PROMPTS_PY
fi
[ -f "$FLOW_GRPO_DIR/$DATASET_REL/train.txt" ] || { log "FATAL: $DATASET_REL/train.txt missing"; exit 1; }

# --- Extra reward functions into flow_grpo/rewards.py (idempotent) ---
REWARDS_PY="$FLOW_GRPO_DIR/flow_grpo/rewards.py"
if ! grep -q "HR_EXTRA_REWARDS" "$REWARDS_PY"; then
  log "Appending extra rewards to rewards.py..."
  cat "$PROJECT_ROOT/scripts/cloud/fg_extra_rewards.py" >> "$REWARDS_PY"
  python3 - "$REWARDS_PY" <<'REG_PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = '        "clipscore": clip_score,\n'
assert anchor in s, "multi_score anchor missing — upstream rewards.py changed"
s = s.replace(anchor, anchor + '        "hpsv2": hpsv2_score,\n        "siglip": siglip_score,\n', 1)
p.write_text(s)
print("registered hpsv2/siglip in multi_score")
REG_PY
fi

# --- Config (idempotent: strip + append) ---
CONFIG_FILE="$FLOW_GRPO_DIR/config/grpo.py"
CFG_NAME="extra_${KEY}_sd3_1gpu"
python3 - "$CONFIG_FILE" "$CFG_NAME" <<'STRIP_PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1]); name = sys.argv[2]; src = p.read_text()
new = re.sub(r"\n\ndef %s\([^)]*\):.*?(?=\n\ndef |\Z)" % re.escape(name), "", src, flags=re.DOTALL)
if new != src: p.write_text(new)
STRIP_PY
cat >> "$CONFIG_FILE" <<PYCONFIG


def ${CFG_NAME}():
    config = general_ocr_sd3_1gpu()
    config.reward_fn = ${REWARD_FN}
    config.dataset = os.path.join(os.getcwd(), "${DATASET_REL}")
    config.save_dir = "logs/${KEY}/sd3.5-M-1gpu-lora"
    _group = int(os.environ.get("FG_GROUP", "2"))
    _nbpe = int(os.environ.get("FG_NBPE", "8"))
    assert _nbpe % 2 == 0, "FG_NBPE must be even (grad-accum = FG_NBPE // 2)"
    config.sample.train_batch_size = _group
    config.sample.num_image_per_prompt = _group
    config.train.batch_size = _group
    config.sample.num_batches_per_epoch = _nbpe
    config.train.gradient_accumulation_steps = _nbpe // 2
    config.train.ema = False
    config.seed = int(os.environ.get("FG_SEED", "42"))
    config.pretrained.model = os.environ.get("FG_MODEL", config.pretrained.model)
    config.train.beta = float(os.environ.get("FG_BETA", str(config.train.beta)))
    config.save_freq = 1
    config.eval_freq = 999
    return config
PYCONFIG

# --- train_sd3.py patches (same as prior runners) ---
TRAIN_PY="$FLOW_GRPO_DIR/scripts/train_sd3.py"
grep -q "epoch > 0 and epoch % config.eval_freq" "$TRAIN_PY" || \
  sed -i 's/if epoch % config.eval_freq == 0:/if epoch > 0 and epoch % config.eval_freq == 0:/' "$TRAIN_PY"
grep -q "torch_dtype=torch.bfloat16" "$TRAIN_PY" || \
  sed -i 's|        config.pretrained.model$|        config.pretrained.model, torch_dtype=torch.bfloat16|' "$TRAIN_PY"
if ! grep -q "LORA_FP32_CAST" "$TRAIN_PY"; then
  python3 - "$TRAIN_PY" <<'CAST_PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); src = p.read_text()
needle = "    transformer = pipeline.transformer\n"
patch = (needle +
    "    # LORA_FP32_CAST: keep LoRA params in fp32 even when base is bf16.\n"
    "    for _p in transformer.parameters():\n"
    "        if _p.requires_grad:\n"
    "            _p.data = _p.data.float()\n")
assert needle in src, "anchor missing — train_sd3.py changed"
p.write_text(src.replace(needle, patch, 1))
CAST_PY
fi

if [ "${FG_SETUP_ONLY:-0}" = "1" ]; then log "setup only — done"; exit 0; fi

# --- Train, capped at WALL_SECONDS ---
mkdir -p "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_${KEY}" "$PROJECT_ROOT/logs"
set +e
timeout --signal=TERM "${WALL_SECONDS}s" \
  accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 \
    --main_process_port 29501 scripts/train_sd3.py --config "config/grpo.py:${CFG_NAME}" \
  2>&1 | tee "$PROJECT_ROOT/logs/flowgrpo_${KEY}_train.log"
RC=$?
set -e
[ $RC -eq 124 ] && log "Wall-clock budget reached." || { [ $RC -ne 0 ] && log "WARN: training rc=$RC"; }
CKPT_DIR="$FLOW_GRPO_DIR/logs/${KEY}/sd3.5-M-1gpu-lora/checkpoints"
if [ -d "$CKPT_DIR" ]; then
  cp -r "$CKPT_DIR"/checkpoint-* "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_${KEY}/" 2>/dev/null || true
  log "copied $(ls "$PROJECT_ROOT/outputs/checkpoints/flowgrpo_${KEY}/" | wc -l) checkpoints"
else
  log "WARN: $CKPT_DIR missing"
fi
log "=== FlowGRPO+$KEY done ==="

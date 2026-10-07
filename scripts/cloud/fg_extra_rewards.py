# --- HR_EXTRA_REWARDS: appended to flow_grpo/flow_grpo/rewards.py by run_flowgrpo_extra.sh ---
# New GRPO rewards backed by the study's own differentiable predictors, so the
# RL arm and the TFG arm optimise literally the same model (one source of truth
# per reward). Interface mirrors clip_score(): fn(images, prompts, metadata) ->
# (scores, {}), images either a [0,1] NCHW tensor or a uint8 NHWC array.


def _hr_predictor_reward(make_pred):
    import os, sys
    import torch
    root = os.environ.get("HR_PROJECT_ROOT", os.path.expanduser("~/haotian_research-1"))
    if root not in sys.path:
        sys.path.insert(0, root)
    pred = make_pred()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            x = images.float()
        else:
            x = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
        x = x.to(pred.device) * 2.0 - 1.0
        with torch.no_grad():
            s = pred(x, list(prompts))
        return [float(v) for v in s.detach().float().cpu()], {}

    return _fn


def hpsv2_score(device):
    def _make():
        from scripts.hpsv2_predictor import HPSv2PromptPredictor
        return HPSv2PromptPredictor(device=device)
    return _hr_predictor_reward(_make)


def siglip_score(device):
    def _make():
        from scripts.siglip_predictor import SiglipPromptPredictor
        return SiglipPromptPredictor(device=device)
    return _hr_predictor_reward(_make)

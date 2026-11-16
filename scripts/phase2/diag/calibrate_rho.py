"""(a) How much of an early-step guidance push survives to the final action? Measured on the toy-Q run
(rho 0.05, same noise and first observation as base50). (b) Calibrate rho for both metrics with the
shipped cross-fitted Qs: action shift in sd units and logit change, last step only vs all five steps."""
import sys, glob, math, numpy as np, torch
sys.path.insert(0, "scripts/phase2")
from q_model import load_q
from guided_sampler import GuidedSampler, GuidanceConfig
np.set_printoptions(precision=3, suppress=True, linewidth=160)
# ---- (a) contraction from the toy run
toy = lambda ctx, x: -(x[..., :7] ** 2).flatten(1).mean(1)
gs = GuidedSampler(None.__class__ and type("M", (), {"sample_actions": None})(), q_fn=toy, cfg=GuidanceConfig(rho=0.05))
rat_sum, rat_last, cosv = [], [], []
for f in sorted(glob.glob("outputs/phase2/dumps/toyq/task*_ep*.npz")):
    zt = np.load(f); zb = np.load(f.replace("toyq", "base50"))
    # inference 0 shares observation and noise with the base run
    assert np.allclose(zt["noise"][0], zb["noise"][0]) and np.allclose(zt["x0_hats"][0, 0], zb["x0_hats"][0, 0], atol=2e-3), "pairing broken"
    xh = torch.tensor(zt["x0_hats"][0].astype(np.float32))          # [steps,H,A] pre-guidance x0_hat of the guided run
    deltas = [gs._guidance_delta({}, xh[s:s + 1])[0][0] for s in range(xh.shape[0])]
    actual = torch.tensor(zt["actions"][0] - zb["actions"][0])[:, :7]
    tot = torch.stack(deltas).sum(0)[:, :7]; last = deltas[-1][:, :7]
    rat_sum.append(float(actual.norm() / tot.norm())); rat_last.append(float(actual.norm() / last.norm()))
    cosv.append(float((actual * tot).sum() / actual.norm() / tot.norm()))
print("(a) toy run, first inference of 3 episodes: |actual shift| / |sum of 5 deltas| =", np.array(rat_sum), " / |last delta| =", np.array(rat_last), " cos(actual, sum) =", np.array(cosv))
# ---- (b) calibration with the shipped Qs
qs = {"hi": load_q("scripts/phase2/q/xfit_hi.pt", "cpu"), "lo": load_q("scripts/phase2/q/xfit_lo.pt", "cpu")}
sd7 = qs["hi"].act_sd.mean(0)
rows = []
for f in sorted(glob.glob("outputs/phase2/dumps/base50/task*_ep*.npz"))[::4]:
    z = np.load(f); q = qs["hi"] if int(z["episode_idx"]) < 25 else qs["lo"]
    rows.append((bool(z["success"]), {"state": torch.tensor(z["state"].astype(np.float32)), "prefix_pool": torch.tensor(z["prefix_pool"].astype(np.float32))},
                 torch.tensor(z["x0_hats"].astype(np.float32)), torch.tensor(z["actions"].astype(np.float32)), q))
def run(metric, rho):
    out = {k: [] for k in ("dl_last", "dl_sum", "sh_last", "sh_sum")}; per = []
    for succ, ctx, xh, act, q in rows:
        g = GuidedSampler(type("M", (), {"sample_actions": None})(), q_fn=q, cfg=GuidanceConfig(rho=rho, metric=metric))
        ds = [g._guidance_delta(ctx, xh[:, s])[0] for s in range(xh.shape[1])]
        tot = torch.stack(ds).sum(0); last = ds[-1]
        with torch.no_grad():
            q0 = q(ctx, act); out["dl_last"].append((q(ctx, act + last) - q0).numpy()); out["dl_sum"].append((q(ctx, act + tot) - q0).numpy())
        out["sh_last"].append((last[:, :, :7] / sd7).pow(2).mean((0, 1)).sqrt().numpy()); out["sh_sum"].append((tot[:, :, :7] / sd7).pow(2).mean((0, 1)).sqrt().numpy())
    c = np.concatenate
    return c(out["dl_last"]).mean(), c(out["dl_sum"]).mean(), np.mean(out["sh_last"], 0), np.mean(out["sh_sum"], 0)
print("(b) shift is rms in units of each dim's natural sd, dims = x y z rx ry rz grip")
for metric, rhos in (("raw", (0.002, 0.005, 0.01)), ("std", (0.02, 0.05, 0.1, 0.2, 0.4))):
    for rho in rhos:
        dl, ds_, sl, ss = run(metric, rho)
        print(f"{metric} rho={rho:<6} dlogit last {dl:6.2f} all5 {ds_:7.2f} | last-step shift/sd {sl} | all5 shift/sd {ss}")

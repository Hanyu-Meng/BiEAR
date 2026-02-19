# train_biear_single_ctrl.py
#
# Same as train_biear.py but uses SINGLE controller:
#   - One shared controller drives Q for BOTH ears
#   - Supports DELTAQ_MODE: "absolute" and "relative"
#   - Config: conf/config_single_ctrl.yaml
#
# Run: python train_biear_single_ctrl.py
#
import os
import json
import re
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model_torch import build_model, build_model_active_single_controller, N_SECTORS, N_DIST_CLASS
from data import DeepEarH5Dataset, DeepEarH5Dataset_Active
from visualize_q import visualize_Q_LR

# =========================
# 1. Config (Load from YAML - single controller config)
# =========================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf", "config_single_ctrl.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ROOT = cfg["ROOT"]
BATCH_SIZE = cfg["BATCH_SIZE"]
EPOCHS = cfg["EPOCHS"]
USE_CC = cfg["USE_CC"]
Active = cfg["Active"]
FIXED_FRONTEND_Q = cfg["FIXED_FRONTEND_Q"]
Controller_Mode = cfg["Controller_Mode"]  # "single"
WEIGHT_DECAY = cfg["WEIGHT_DECAY"]
GRAD_CLIP_NORM = cfg["GRAD_CLIP_NORM"]
ALPHA = cfg["ALPHA"]
LR_FB = cfg["LR_FB"]
LR_BACKEND = cfg["LR_BACKEND"]
REG_Q_W = cfg["REG_Q_W"]
REG_SMOOTH_W = cfg["REG_SMOOTH_W"]
FREEZE_Q_CONTROLLER_ONLY = cfg["FREEZE_Q_CONTROLLER_ONLY"]

DELTAQ_BASE = cfg["DELTAQ_BASE"]
DELTAQ_LOW_FACTOR = cfg["DELTAQ_LOW_FACTOR"]
DELTAQ_HIGH_FACTOR = cfg["DELTAQ_HIGH_FACTOR"]
DELTAQ_MODE = cfg.get("DELTAQ_MODE", "absolute")

LOSS_WEIGHT_SOUND = cfg["LOSS_WEIGHT_SOUND"]
LOSS_WEIGHT_AOA = cfg["LOSS_WEIGHT_AOA"]
LOSS_WEIGHT_DIST = cfg["LOSS_WEIGHT_DIST"]

HIST_EVERY = cfg["HIST_EVERY"]
MAX_PARAM_LOG = cfg["MAX_PARAM_LOG"]
PRINT_EVERY = cfg["PRINT_EVERY"]
SAVE_EVERY_EPOCH = cfg["SAVE_EVERY_EPOCH"]
type = "fixedQ" if FIXED_FRONTEND_Q else "adaptiveQ"
COMMENTS = cfg.get("COMMENTS", "")
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
RUNS_ROOT = cfg["RUNS_ROOT"]

def _slug(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "-", x)
    x = re.sub(r"[^a-z0-9_\-\.]+", "", x)
    return x[:120]

exp_name = "_".join([
    f"{'active' if Active else 'passive'}",
    "ctrl-single",
    f"fixedq-{int(bool(FIXED_FRONTEND_Q) and bool(Active))}",
    f"type-{type}",
    f"alpha{ALPHA:g}",
    f"cc-{int(USE_CC)}",
    f"qctrlfrozen-{int(bool(FREEZE_Q_CONTROLLER_ONLY) and bool(Active))}",
    f"bs{BATCH_SIZE}",
    f"lrfb{LR_FB:g}",
    f"lrbe{LR_BACKEND:g}",
    f"wd{WEIGHT_DECAY:g}",
    f"lossw{LOSS_WEIGHT_SOUND:.2f}_{LOSS_WEIGHT_AOA:.2f}_{LOSS_WEIGHT_DIST:.2f}",
    f"run{run_id}",
    f"dq{DELTAQ_BASE:g}_lo{DELTAQ_LOW_FACTOR:g}_hi{DELTAQ_HIGH_FACTOR:g}_{DELTAQ_MODE[:3]}",
    f"{_slug(COMMENTS)}" if COMMENTS else "",
])

RUN_DIR = os.path.join(RUNS_ROOT, exp_name)
TB_DIR = os.path.join(RUN_DIR, "tb")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
JSON_DIR = os.path.join(RUN_DIR, "logs_json")
VIS_DIR = os.path.join(RUN_DIR, "q_vis")
META_DIR = os.path.join(RUN_DIR, "meta")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

writer = SummaryWriter(TB_DIR)
CKPT_BEST = os.path.join(CKPT_DIR, "best.pth")
CKPT_LAST = os.path.join(CKPT_DIR, "last.pth")
HIST_PATH = os.path.join(JSON_DIR, "history.json")
SETTINGS_PATH = os.path.join(META_DIR, "settings.json")
global_step = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("[Run dir]:", RUN_DIR)
print("[Controller] Single (one controller for both ears)")
print("[DELTAQ_MODE]:", DELTAQ_MODE)

settings = dict(
    ROOT=ROOT, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, USE_CC=USE_CC, Active=Active,
    FIXED_FRONTEND_Q=bool(FIXED_FRONTEND_Q), Controller_Mode="single", ALPHA=ALPHA,
    WEIGHT_DECAY=WEIGHT_DECAY, GRAD_CLIP_NORM=GRAD_CLIP_NORM, LR_FB=LR_FB, LR_BACKEND=LR_BACKEND,
    REG_Q_W=REG_Q_W, REG_SMOOTH_W=REG_SMOOTH_W, FREEZE_Q_CONTROLLER_ONLY=bool(FREEZE_Q_CONTROLLER_ONLY),
    LOSS_WEIGHT_SOUND=LOSS_WEIGHT_SOUND, LOSS_WEIGHT_AOA=LOSS_WEIGHT_AOA, LOSS_WEIGHT_DIST=LOSS_WEIGHT_DIST,
    run_id=run_id, exp_name=exp_name,
    DELTAQ_BASE=DELTAQ_BASE, DELTAQ_LOW_FACTOR=DELTAQ_LOW_FACTOR, DELTAQ_HIGH_FACTOR=DELTAQ_HIGH_FACTOR,
    DELTAQ_MODE=DELTAQ_MODE, comments=COMMENTS
)
with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=2)

# =========================
# 2. Paths
# =========================
if Active:
    train_h5 = f"{ROOT}/anechoic_train_active_wav.h5"
    val_h5 = f"{ROOT}/anechoic_val_active_wav.h5"
    test_h5 = f"{ROOT}/anechoic_test1_active_wav.h5"
else:
    train_h5 = f"{ROOT}/anechoic_train_gt_group_phase.h5"
    val_h5 = f"{ROOT}/anechoic_val_gt_group_phase.h5"
    test_h5 = f"{ROOT}/anechoic_test2_gt_group_phase.h5"

# =========================
# 3. Helpers
# =========================
def _grad_norm(params, norm_type=2.0):
    total_norm = 0.0
    has_nonfinite = False
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.numel() == 0:
            continue
        if not torch.isfinite(g).all():
            has_nonfinite = True
        n = g.norm(norm_type)
        if torch.isfinite(n):
            total_norm += float(n.item() ** norm_type)
        else:
            has_nonfinite = True
    total_norm = total_norm ** (1.0 / norm_type) if total_norm > 0 else 0.0
    return total_norm, has_nonfinite

def log_grads_tensorboard(model, step, fb_params=None, backend_params=None, hist_every=50, max_param_log=200):
    all_params = [p for p in model.parameters() if p.requires_grad]
    total_norm, bad_all = _grad_norm(all_params)
    writer.add_scalar("grad/total_norm", total_norm, step)
    writer.add_scalar("grad/has_nonfinite", float(bad_all), step)
    if fb_params:
        fb_norm, bad_fb = _grad_norm(fb_params)
        writer.add_scalar("grad/fb_norm", fb_norm, step)
        writer.add_scalar("grad/fb_has_nonfinite", float(bad_fb), step)
    if backend_params:
        be_norm, bad_be = _grad_norm(backend_params)
        writer.add_scalar("grad/backend_norm", be_norm, step)
        writer.add_scalar("grad/backend_has_nonfinite", float(bad_be), step)
    if step % hist_every == 0:
        cnt = 0
        for name, p in model.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            g = p.grad.detach()
            if g.numel() == 0:
                continue
            if not torch.isfinite(g).all():
                writer.add_scalar(f"grad_bad/{name}", 1.0, step)
                continue
            if g.abs().max().item() == 0.0:
                continue
            writer.add_histogram(f"grad_hist/{name}", g.float().cpu(), step)
            cnt += 1
            if cnt >= max_param_log:
                break
    return total_norm, bad_all

def unpack_targets(y_batch):
    B = y_batch.shape[0]
    y_sound = torch.zeros(B, N_SECTORS, device=y_batch.device)
    y_aoa = torch.zeros(B, N_SECTORS, device=y_batch.device)
    y_dist = torch.zeros(B, N_SECTORS, N_DIST_CLASS, device=y_batch.device)
    for k in range(N_SECTORS):
        base = k * 7
        y_sound[:, k] = y_batch[:, base]
        y_aoa[:, k] = y_batch[:, base + 1]
        y_dist[:, k, :] = y_batch[:, base + 2: base + 7]
    return y_sound, y_aoa, y_dist

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def freeze_q_controller_only_single(model):
    """Single controller: freeze bifb.q_rnn and bifb.q_out (at bifb level)."""
    if not hasattr(model, "bifb") or model.bifb is None:
        return []
    bifb = model.bifb
    frozen_names = []
    for mod_name in ["q_rnn", "q_out"]:
        mod = getattr(bifb, mod_name, None)
        if mod is not None:
            for pn, p in mod.named_parameters():
                p.requires_grad = False
                frozen_names.append(f"bifb.{mod_name}.{pn}")
    return frozen_names

@torch.no_grad()
def sanity_debug_one_batch(model, loader):
    batch = next(iter(loader))
    model.eval()
    if Active:
        wavL, wavR, x3, y = batch
        wavL, wavR, x3, y = wavL.to(device), wavR.to(device), x3.to(device), y.to(device)
        sound_logits, aoa_pred, dist_logits = model(wavL.float(), wavR.float(), x3.float())
    else:
        x1, x2, x3, x4, x5, y = batch
        x1, x2, x3, x4, x5, y = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device), y.to(device)
        sound_logits, aoa_pred, dist_logits = model(x1, x2, x3, x4, x5)
    print("[Sanity] logits finite:", torch.isfinite(sound_logits).all().item(),
          torch.isfinite(aoa_pred).all().item(), torch.isfinite(dist_logits).all().item())

def _sanitize_x3(x3):
    x3 = x3.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    x3 = x3 / torch.clamp(x3.abs().amax(dim=1, keepdim=True), min=1.0)
    return torch.clamp(x3, -5.0, 5.0)

def _is_better_tuple(curr, best, eps=1e-12):
    if best is None:
        return True
    cs, ca, cd = curr
    bs, ba, bd = best
    if cs > bs + eps:
        return True
    if abs(cs - bs) <= eps and ca < ba - eps:
        return True
    if abs(cs - bs) <= eps and abs(ca - ba) <= eps and cd > bd + eps:
        return True
    return False

# =========================
# 4. Data
# =========================
if Active:
    train_ds = DeepEarH5Dataset_Active(train_h5)
    val_ds = DeepEarH5Dataset_Active(val_h5)
    test_ds = DeepEarH5Dataset_Active(test_h5)
else:
    train_ds = DeepEarH5Dataset(train_h5)
    val_ds = DeepEarH5Dataset(val_h5)
    test_ds = DeepEarH5Dataset(test_h5)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# =========================
# 5. Model & losses
# =========================
model = (
    build_model_active_single_controller(
        use_cc=USE_CC, fb_alpha=ALPHA, fixed_frontend_q=bool(FIXED_FRONTEND_Q),
        deltaQ_base=DELTAQ_BASE, deltaQ_low_factor=DELTAQ_LOW_FACTOR,
        deltaQ_high_factor=DELTAQ_HIGH_FACTOR, deltaQ_mode=DELTAQ_MODE,
    ) if Active else build_model(use_cc=USE_CC)
).to(device)

def get_frontend_backend_params(model):
    fb_params_all = list(model.bifb.parameters()) if (hasattr(model, "bifb") and model.bifb is not None) else []
    fb_params_trainable = [p for p in fb_params_all if p.requires_grad]
    fb_ids = set(id(p) for p in fb_params_trainable)
    be_params = [p for p in model.parameters() if p.requires_grad and id(p) not in fb_ids]
    return fb_params_trainable, be_params

if Active and FREEZE_Q_CONTROLLER_ONLY and (not FIXED_FRONTEND_Q):
    frozen = freeze_q_controller_only_single(model)
    print(f"✅ Frozen Q-controller (single): {len(frozen)} tensors")

total_p = count_total_params(model)
trainable_p = count_trainable_params(model)
print(f"[Params] total={total_p:,} | trainable={trainable_p:,} ({100*trainable_p/max(total_p,1):.1f}%)")

fb_params, backend_params = (None, None)
if Active:
    fb_params, backend_params = get_frontend_backend_params(model)
    print(f"[Frontend trainable] {sum(p.numel() for p in fb_params)} | [Backend] {sum(p.numel() for p in backend_params)}")

pos_weight = torch.full((N_SECTORS,), 3.0, device=device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
aoa_loss_fn = nn.SmoothL1Loss(beta=0.02)
ce = nn.CrossEntropyLoss()

def compute_task_loss(sound_logits, aoa_pred, dist_logits, y):
    y_sound, y_aoa, y_dist = unpack_targets(y)
    dist_target = y_dist.argmax(dim=-1).reshape(-1)
    dist_pred_logits = dist_logits.reshape(-1, N_DIST_CLASS)
    loss = LOSS_WEIGHT_SOUND * bce(sound_logits, y_sound) + LOSS_WEIGHT_AOA * aoa_loss_fn(aoa_pred, y_aoa) + LOSS_WEIGHT_DIST * ce(dist_pred_logits, dist_target)
    with torch.no_grad():
        metrics = dict(
            sound_acc=((torch.sigmoid(sound_logits) > 0.5) == y_sound).float().mean().item(),
            aoa_mae=(aoa_pred - y_aoa).abs().mean().item(),
            dist_acc=(dist_pred_logits.argmax(dim=-1) == dist_target).float().mean().item(),
        )
    metrics["loss"] = float(loss.item())
    return loss, metrics

def compute_loss_passive(batch):
    x1, x2, x3, x4, x5, y = batch
    x1, x2, x3, x4, x5, y = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device), y.to(device)
    sound_logits, aoa_pred, dist_logits = model(x1, x2, x3, x4, x5)
    return compute_task_loss(sound_logits, aoa_pred, dist_logits, y)

def compute_loss_active(batch):
    wavL, wavR, x3, y = batch
    wavL, wavR, x3 = wavL.to(device).float(), wavR.to(device).float(), _sanitize_x3(x3.to(device))
    y = y.to(device).float()
    if wavL.abs().max() > 2.0 or wavR.abs().max() > 2.0:
        wavL, wavR = wavL / 32768.0, wavR / 32768.0
    wavL, wavR = torch.clamp(wavL, -1.0, 1.0), torch.clamp(wavR, -1.0, 1.0)
    sound_logits, aoa_pred, dist_logits = model(wavL, wavR, x3)
    task_loss, metrics = compute_task_loss(sound_logits, aoa_pred, dist_logits, y)
    Q = getattr(model, "last_Q", None)
    if Q is not None and hasattr(model, "bifb") and hasattr(model.bifb, "Q0"):
        Q0 = model.bifb.Q0.view(1, 1, -1)
        logQ = torch.log(Q + 1e-8)
        logQ0 = torch.log(Q0 + 1e-8)
        reg_q = ((logQ - logQ0) ** 2).mean()
        reg_smooth = ((logQ[:, :, 1:] - logQ[:, :, :-1]) ** 2).mean()
        loss = task_loss + REG_Q_W * reg_q + REG_SMOOTH_W * reg_smooth
        metrics["loss"] = float(loss.item())
        return loss, metrics
    return task_loss, metrics

# =========================
# 6. Train loop
# =========================
def run_epoch(loader, optimizer=None, train=True, stage="train", epoch_idx=0):
    global global_step
    model.train() if train else model.eval()
    total, sum_loss, sum_sound_acc, sum_aoa_mae, sum_dist_acc, skipped = 0, 0.0, 0.0, 0.0, 0.0, 0
    with (torch.enable_grad() if train else torch.no_grad()):
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss_active(batch) if Active else compute_loss_passive(batch)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            if train:
                loss.backward()
                if Active and fb_params is not None and backend_params is not None and len(fb_params) > 0:
                    torch.nn.utils.clip_grad_norm_(fb_params, 0.2)
                    torch.nn.utils.clip_grad_norm_(backend_params, 3.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                writer.add_scalar(f"{stage}/loss_step", metrics["loss"], global_step)
                writer.add_scalar(f"{stage}/sound_acc_step", metrics["sound_acc"], global_step)
                writer.add_scalar(f"{stage}/aoa_mae_step", metrics["aoa_mae"], global_step)
                writer.add_scalar(f"{stage}/dist_acc_step", metrics["dist_acc"], global_step)
                log_grads_tensorboard(model, global_step, fb_params, backend_params, HIST_EVERY, MAX_PARAM_LOG)
                if global_step % PRINT_EVERY == 0:
                    print(f"[{global_step:06d}] loss={metrics['loss']:.4f} sound={metrics['sound_acc']:.3f} aoa_mae={metrics['aoa_mae']:.3f} dist={metrics['dist_acc']:.3f}")
                optimizer.step()
                global_step += 1
            bs = batch[0].shape[0]
            total += bs
            sum_loss += metrics["loss"] * bs
            sum_sound_acc += metrics["sound_acc"] * bs
            sum_aoa_mae += metrics["aoa_mae"] * bs
            sum_dist_acc += metrics["dist_acc"] * bs
    out = dict(loss=sum_loss/total if total else float("nan"), sound_acc=sum_sound_acc/total if total else 0.0,
               aoa_mae=sum_aoa_mae/total if total else float("nan"), dist_acc=sum_dist_acc/total if total else 0.0, skipped=skipped)
    writer.add_scalar(f"{stage}/loss_epoch", out["loss"], epoch_idx)
    writer.add_scalar(f"{stage}/sound_acc_epoch", out["sound_acc"], epoch_idx)
    writer.add_scalar(f"{stage}/aoa_mae_epoch", out["aoa_mae"], epoch_idx)
    writer.add_scalar(f"{stage}/dist_acc_epoch", out["dist_acc"], epoch_idx)
    return out

# =========================
# 7. Training
# =========================
sanity_debug_one_batch(model, train_loader)
if Active and fb_params and len(fb_params) > 0:
    optimizer = torch.optim.Adam(
        [{"params": fb_params, "lr": LR_FB}, {"params": backend_params, "lr": LR_BACKEND}],
        weight_decay=WEIGHT_DECAY, eps=1e-7
    )
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BACKEND, weight_decay=WEIGHT_DECAY, eps=1e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

best_val_loss = float("inf")
best_val_tuple = None
history = {"train": [], "val": []}

for e in range(1, EPOCHS + 1):
    tr = run_epoch(train_loader, optimizer, train=True, stage="train", epoch_idx=e)
    va = run_epoch(val_loader, None, train=False, stage="val", epoch_idx=e)
    history["train"].append(tr)
    history["val"].append(va)
    print(f"[{e:03d}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_sound={va['sound_acc']:.3f} val_aoa_mae={va['aoa_mae']:.3f} val_dist={va['dist_acc']:.3f}")
    if torch.isfinite(torch.tensor(va["loss"])) and va["loss"] < best_val_loss:
        best_val_loss = va["loss"]
    scheduler.step(va["loss"])
    curr = (va["sound_acc"], va["aoa_mae"], va["dist_acc"])
    if _is_better_tuple(curr, best_val_tuple):
        best_val_tuple = curr
        torch.save(model.state_dict(), CKPT_BEST)
        print(f"  🔥 Saved best: sound={curr[0]:.4f} aoa_mae={curr[1]:.4f} dist={curr[2]:.4f}")
    if SAVE_EVERY_EPOCH:
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"epoch{e:03d}.pth"))

torch.save(model.state_dict(), CKPT_LAST)
with open(HIST_PATH, "w") as f:
    json.dump(history, f, indent=2)
print("Training finished.")

# =========================
# 8. Test + Q viz
# =========================
if os.path.exists(CKPT_BEST):
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device))
model.eval()
te = run_epoch(test_loader, None, train=False, stage="test", epoch_idx=0)
print("Test metrics:", te)
with open(os.path.join(JSON_DIR, "test_metrics.json"), "w") as f:
    json.dump(te, f, indent=2)

if Active:
    visualize_Q_LR(model=model, dataloader=test_loader, device=device, save_dir=VIS_DIR, max_batches=5, sample_per_batch=1)

writer.close()
print("Done.")

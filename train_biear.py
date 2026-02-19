# train_torch.py (ONE-STAGE JOINT TRAINING + TensorBoard + structured run folders)
import os
import json
import re
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model_torch import build_model, build_model_active, N_SECTORS, N_DIST_CLASS
from data import DeepEarH5Dataset, DeepEarH5Dataset_Active
from visualize_q import visualize_Q_LR

# =========================
# 1. Config (Load from YAML)
# =========================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ROOT = cfg["ROOT"]
BATCH_SIZE = cfg["BATCH_SIZE"]
EPOCHS = cfg["EPOCHS"]
USE_CC = cfg["USE_CC"]

# IMPORTANT:
# Active=False -> passive model (uses precomputed features in H5)
# Active=True  -> waveform -> (bi)filterbank -> network
Active = cfg["Active"]

# ✅ NEW: fixed front-end (Q always equals Q0, no adaptation)
FIXED_FRONTEND_Q = cfg["FIXED_FRONTEND_Q"]

# Dual-only now (single removed)
Controller_Mode = cfg["Controller_Mode"]

WEIGHT_DECAY = cfg["WEIGHT_DECAY"]
GRAD_CLIP_NORM = cfg["GRAD_CLIP_NORM"]

ALPHA = cfg["ALPHA"]  # feedback alpha for active model (irrelevant when FIXED_FRONTEND_Q=True)

# ---- Joint training LR (param groups) ----
LR_FB = cfg["LR_FB"]
LR_BACKEND = cfg["LR_BACKEND"]

# Q regularization (optional)
REG_Q_W = cfg["REG_Q_W"]
REG_SMOOTH_W = cfg["REG_SMOOTH_W"]

# Freeze only Q-controller (q_rnn + q_out), keep everything else learnable
#  even if frozen, Q can still vary (depends on inputs). For truly fixed Q, use FIXED_FRONTEND_Q=True.
FREEZE_Q_CONTROLLER_ONLY = cfg["FREEZE_Q_CONTROLLER_ONLY"]  # set True/False

DELTAQ_BASE = cfg["DELTAQ_BASE"]          # overall +/- Q range scale
DELTAQ_LOW_FACTOR = cfg["DELTAQ_LOW_FACTOR"]    # low freq bands: smaller change
DELTAQ_HIGH_FACTOR = cfg["DELTAQ_HIGH_FACTOR"]   # high freq bands: larger change
DELTAQ_MODE = cfg.get("DELTAQ_MODE", "absolute")  # "absolute" or "relative"

# Loss weights
LOSS_WEIGHT_SOUND = cfg["LOSS_WEIGHT_SOUND"]
LOSS_WEIGHT_AOA = cfg["LOSS_WEIGHT_AOA"]
LOSS_WEIGHT_DIST = cfg["LOSS_WEIGHT_DIST"]

# =========================
# TensorBoard / Logging
# =========================
HIST_EVERY = cfg["HIST_EVERY"]
MAX_PARAM_LOG = cfg["MAX_PARAM_LOG"]
PRINT_EVERY = cfg["PRINT_EVERY"]
SAVE_EVERY_EPOCH = cfg["SAVE_EVERY_EPOCH"]
type = "fixedQ" if FIXED_FRONTEND_Q else "adaptiveQ"
COMMENTS = cfg["COMMENTS"]
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

RUNS_ROOT = cfg["RUNS_ROOT"]  # change to any parent folder you want

def _slug(x: str) -> str:
    x = str(x)
    x = x.strip().lower()
    x = re.sub(r"\s+", "-", x)
    x = re.sub(r"[^a-z0-9_\-\.]+", "", x)
    return x[:120]

exp_name = "_".join([
    f"{'active' if Active else 'passive'}",
    "ctrl-dual",
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

RUN_DIR   = os.path.join(RUNS_ROOT, exp_name)
TB_DIR    = os.path.join(RUN_DIR, "tb")
CKPT_DIR  = os.path.join(RUN_DIR, "checkpoints")
JSON_DIR  = os.path.join(RUN_DIR, "logs_json")
VIS_DIR   = os.path.join(RUN_DIR, "q_vis")
META_DIR  = os.path.join(RUN_DIR, "meta")

os.makedirs(RUN_DIR,  exist_ok=True)
os.makedirs(TB_DIR,   exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)
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
print("[TensorBoard] logdir:", TB_DIR)
print("[Save paths] best:", CKPT_BEST)
print("[Save paths] last:", CKPT_LAST)
print("[Save paths] hist:", HIST_PATH)

settings = dict(
    ROOT=ROOT,
    BATCH_SIZE=BATCH_SIZE,
    EPOCHS=EPOCHS,
    USE_CC=USE_CC,
    Active=Active,
    FIXED_FRONTEND_Q=bool(FIXED_FRONTEND_Q),
    Controller_Mode="dual",
    ALPHA=ALPHA,
    WEIGHT_DECAY=WEIGHT_DECAY,
    GRAD_CLIP_NORM=GRAD_CLIP_NORM,
    LR_FB=LR_FB,
    LR_BACKEND=LR_BACKEND,
    REG_Q_W=REG_Q_W,
    REG_SMOOTH_W=REG_SMOOTH_W,
    FREEZE_Q_CONTROLLER_ONLY=bool(FREEZE_Q_CONTROLLER_ONLY),
    LOSS_WEIGHT_SOUND=LOSS_WEIGHT_SOUND,
    LOSS_WEIGHT_AOA=LOSS_WEIGHT_AOA,
    LOSS_WEIGHT_DIST=LOSS_WEIGHT_DIST,
    run_id=run_id,
    exp_name=exp_name,
    DELTAQ_BASE=DELTAQ_BASE,
    DELTAQ_LOW_FACTOR=DELTAQ_LOW_FACTOR,
    DELTAQ_HIGH_FACTOR=DELTAQ_HIGH_FACTOR,
    DELTAQ_MODE=DELTAQ_MODE,
    comments=COMMENTS
)

with open(SETTINGS_PATH, "w") as f:
    json.dump(settings, f, indent=2)

# =========================
# 2. Paths
# =========================
if Active:
    train_h5 = f"{ROOT}/anechoic_train_active_wav.h5"
    val_h5   = f"{ROOT}/anechoic_val_active_wav.h5"
    test_h5  = f"{ROOT}/anechoic_test1_active_wav.h5"
else:
    train_h5 = f"{ROOT}/anechoic_train_gt_group_phase.h5"
    val_h5   = f"{ROOT}/anechoic_val_gt_group_phase.h5"
    test_h5  = f"{ROOT}/anechoic_test2_gt_group_phase.h5"

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

def log_grads_tensorboard(model, step, fb_params=None, backend_params=None,
                          hist_every=50, max_param_log=200):
    all_params = [p for p in model.parameters() if p.requires_grad]
    total_norm, bad_all = _grad_norm(all_params)
    writer.add_scalar("grad/total_norm", total_norm, step)
    writer.add_scalar("grad/has_nonfinite", float(bad_all), step)

    if fb_params is not None:
        fb_norm, bad_fb = _grad_norm(fb_params)
        writer.add_scalar("grad/fb_norm", fb_norm, step)
        writer.add_scalar("grad/fb_has_nonfinite", float(bad_fb), step)

    if backend_params is not None:
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

def unpack_targets(y_batch: torch.Tensor):
    B = y_batch.shape[0]
    y_sound = torch.zeros(B, N_SECTORS, device=y_batch.device)
    y_aoa   = torch.zeros(B, N_SECTORS, device=y_batch.device)
    y_dist  = torch.zeros(B, N_SECTORS, N_DIST_CLASS, device=y_batch.device)

    for k in range(N_SECTORS):
        base = k * 7
        y_sound[:, k] = y_batch[:, base]
        y_aoa[:, k]   = y_batch[:, base + 1]
        if N_DIST_CLASS == 5:
            y_dist[:, k, :] = y_batch[:, base + 2: base + 7]
        elif N_DIST_CLASS == 4:
            y_dist[:, k, :] = y_batch[:, base + 2: base + 6]
        else:
            raise ValueError(f"Unexpected N_DIST_CLASS = {N_DIST_CLASS}")

    return y_sound, y_aoa, y_dist

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

def freeze_q_controller_only(model):
    """
    Dual-only now:
      freeze model.bifb.fb_L / fb_R controller modules (q_rnn + q_out)
    """
    if not hasattr(model, "bifb"):
        print("[freeze_q_controller_only] No model.bifb found. Skip.")
        return []

    frozen_names = []
    bifb = model.bifb

    for ear in ["fb_L", "fb_R"]:
        fb = getattr(bifb, ear, None)
        if fb is None:
            continue
        for mod_name in ["q_rnn", "q_out"]:
            if hasattr(fb, mod_name):
                mod = getattr(fb, mod_name)
                for pn, p in mod.named_parameters():
                    p.requires_grad = False
                    frozen_names.append(f"bifb.{ear}.{mod_name}.{pn}")

    return frozen_names

@torch.no_grad()
def sanity_debug_one_batch(model, loader):
    print("\n[Sanity] Debug one batch BEFORE training ...")
    batch = next(iter(loader))
    model.eval()

    if Active:
        wavL, wavR, x3, y = batch
        wavL, wavR, x3, y = wavL.to(device), wavR.to(device), x3.to(device), y.to(device)
        print("[debug] wavL min/max:", wavL.min().item(), wavL.max().item())
        print("[debug] wavR min/max:", wavR.min().item(), wavR.max().item())
        sound_logits, aoa_pred, dist_logits = model(wavL.float(), wavR.float(), x3.float())
        print("[debug] logits finite:",
              torch.isfinite(sound_logits).all().item(),
              torch.isfinite(aoa_pred).all().item(),
              torch.isfinite(dist_logits).all().item())
        Q = getattr(model, "last_Q", None)
        if Q is not None:
            print("[debug] Q shape:", tuple(Q.shape),
                  "min/max:", Q.min().item(), Q.max().item(),
                  "finite:", torch.isfinite(Q).all().item())
    else:
        x1, x2, x3, x4, x5, y = batch
        x1, x2, x3, x4, x5, y = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device), y.to(device)
        sound_logits, aoa_pred, dist_logits = model(x1, x2, x3, x4, x5)
        print("[debug] logits finite:",
              torch.isfinite(sound_logits).all().item(),
              torch.isfinite(aoa_pred).all().item(),
              torch.isfinite(dist_logits).all().item())

def _sanitize_x3(x3: torch.Tensor) -> torch.Tensor:
    x3 = x3.float()
    x3 = torch.nan_to_num(x3, nan=0.0, posinf=0.0, neginf=0.0)
    maxabs = x3.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(maxabs, min=1.0)
    x3 = x3 / scale
    x3 = torch.clamp(x3, -5.0, 5.0)
    return x3

def _is_better_tuple(curr, best, eps=1e-12):
    if best is None:
        return True
    cs, ca, cd = curr
    bs, ba, bd = best
    if cs > bs + eps:
        return True
    if abs(cs - bs) <= eps:
        if ca < ba - eps:
            return True
        if abs(ca - ba) <= eps:
            if cd > bd + eps:
                return True
    return False

# =========================
# 4. Data
# =========================
if Active:
    train_ds = DeepEarH5Dataset_Active(train_h5)
    val_ds   = DeepEarH5Dataset_Active(val_h5)
    test_ds  = DeepEarH5Dataset_Active(test_h5)
else:
    train_ds = DeepEarH5Dataset(train_h5)
    val_ds   = DeepEarH5Dataset(val_h5)
    test_ds  = DeepEarH5Dataset(test_h5)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# =========================
# 5. Model & losses
# =========================
model = (
    build_model_active(
        use_cc=USE_CC,
        fb_alpha=ALPHA,
        fixed_frontend_q=bool(FIXED_FRONTEND_Q),
        deltaQ_base=DELTAQ_BASE,
        deltaQ_low_factor=DELTAQ_LOW_FACTOR,
        deltaQ_high_factor=DELTAQ_HIGH_FACTOR,
        deltaQ_mode=DELTAQ_MODE,
    ) if Active else build_model(use_cc=USE_CC)
).to(device)

print(model)

def get_frontend_backend_params(model):
    """
    Returns:
      fb_params_trainable: list[Parameter]   # trainable params in frontend
      be_params_trainable: list[Parameter]   # trainable params in backend
    Notes:
      - If FIXED_FRONTEND_Q=True (and you used the "no-controller" FB implementation),
        fb_params_trainable should be EMPTY.
    """
    fb_params_all = []
    if hasattr(model, "bifb") and model.bifb is not None:
        # Collect any parameters that live inside bifb (both ears)
        # This is safe even if bifb has no parameters.
        fb_params_all = list(model.bifb.parameters())

    fb_params_trainable = [p for p in fb_params_all if p.requires_grad]
    fb_ids = set(id(p) for p in fb_params_trainable)

    be_params_trainable = [p for p in model.parameters() if p.requires_grad and (id(p) not in fb_ids)]
    return fb_params_trainable, be_params_trainable

# ✅ Optional: freeze controller weights (only meaningful for adaptive frontend)
frozen_qctrl = []
if Active and FREEZE_Q_CONTROLLER_ONLY and (not FIXED_FRONTEND_Q):
    frozen_qctrl = freeze_q_controller_only(model)
    print(f"✅ Frozen Q-controller params: {len(frozen_qctrl)} tensors")
    if len(frozen_qctrl) > 0:
        for s in frozen_qctrl[:10]:
            print("   -", s)
elif Active and FREEZE_Q_CONTROLLER_ONLY and FIXED_FRONTEND_Q:
    print("⚠️ FREEZE_Q_CONTROLLER_ONLY=True but FIXED_FRONTEND_Q=True -> no controller exists, skip freezing.")

# ✅ IMPORTANT: If FIXED_FRONTEND_Q=True, DO NOT force model.bifb.fb_L.freeze_Q here.
# Because the fixed frontend implementation has no controller and may not have freeze_Q attribute.
if Active and FIXED_FRONTEND_Q:
    print("✅ Fixed front-end enabled (no controller modules should exist).")

total_p = count_total_params(model)
trainable_p = count_trainable_params(model)
ratio = 100.0 * trainable_p / max(total_p, 1)
print(f"[Params] total={total_p:,} | trainable={trainable_p:,} ({ratio:.1f}%)")

fb_params, backend_params = (None, None)
if Active:
    fb_params, backend_params = get_frontend_backend_params(model)

    print(f"[Frontend trainable tensors] {len(fb_params)}  | numel={sum(p.numel() for p in fb_params)}")
    print(f"[Backend  trainable tensors] {len(backend_params)} | numel={sum(p.numel() for p in backend_params)}")

    # Optional safety check: when FIXED_FRONTEND_Q=True, fb_params should be empty
    if FIXED_FRONTEND_Q and (sum(p.numel() for p in fb_params) != 0):
        print("⚠️ WARNING: FIXED_FRONTEND_Q=True but frontend still has trainable params. "
              "This means your model_torch fixed FB still contains learnable modules.")
else:
    fb_params = None
    backend_params = None

pos_weight = torch.full((N_SECTORS,), 3.0, device=device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
aoa_loss_fn = nn.SmoothL1Loss(beta=0.02)
ce  = nn.CrossEntropyLoss()

def compute_task_loss(sound_logits, aoa_pred, dist_logits, y):
    y_sound, y_aoa, y_dist = unpack_targets(y)
    dist_target = y_dist.argmax(dim=-1).reshape(-1)
    dist_pred_logits = dist_logits.reshape(-1, N_DIST_CLASS)

    loss_sound = bce(sound_logits, y_sound)
    loss_aoa   = aoa_loss_fn(aoa_pred, y_aoa)
    loss_dist  = ce(dist_pred_logits, dist_target)

    # loss = 0.4 * loss_sound + 0.35 * loss_aoa + 0.25 * loss_dist
    loss = LOSS_WEIGHT_SOUND * loss_sound + LOSS_WEIGHT_AOA * loss_aoa + LOSS_WEIGHT_DIST * loss_dist

    with torch.no_grad():
        sound_acc = ((torch.sigmoid(sound_logits) > 0.5) == y_sound).float().mean().item()
        aoa_mae   = (aoa_pred - y_aoa).abs().mean().item()
        dist_acc  = (dist_pred_logits.argmax(dim=-1) == dist_target).float().mean().item()

    return loss, {
        "loss": float(loss.item()),
        "sound_acc": sound_acc,
        "aoa_mae": aoa_mae,
        "dist_acc": dist_acc,
    }

def compute_loss_passive(batch):
    x1, x2, x3, x4, x5, y = batch
    x1 = x1.to(device)
    x2 = x2.to(device)
    x3 = x3.to(device)
    x4 = x4.to(device)
    x5 = x5.to(device)
    y  = y.to(device)
    sound_logits, aoa_pred, dist_logits = model(x1, x2, x3, x4, x5)
    return compute_task_loss(sound_logits, aoa_pred, dist_logits, y)

def compute_loss_active(batch):
    wavL, wavR, x3, y = batch
    wavL = wavL.to(device).float()
    wavR = wavR.to(device).float()
    x3   = _sanitize_x3(x3.to(device))
    y    = y.to(device).float()

    maxabsL = wavL.abs().max().item()
    maxabsR = wavR.abs().max().item()
    if maxabsL > 2.0 or maxabsR > 2.0:
        wavL = wavL / 32768.0
        wavR = wavR / 32768.0

    wavL = torch.clamp(wavL, -1.0, 1.0)
    wavR = torch.clamp(wavR, -1.0, 1.0)

    sound_logits, aoa_pred, dist_logits = model(wavL, wavR, x3)
    task_loss, metrics = compute_task_loss(sound_logits, aoa_pred, dist_logits, y)

    # If fixed Q, reg terms are always ~0; keep code but harmless
    Q = getattr(model, "last_Q", None)
    if Q is None:
        return task_loss, metrics

    if not hasattr(model, "bifb") or not hasattr(model.bifb, "Q0"):
        return task_loss, metrics

    Q0 = model.bifb.Q0.view(1, 1, -1)
    logQ = torch.log(Q + 1e-8)
    logQ0 = torch.log(Q0 + 1e-8)

    reg_q = ((logQ - logQ0) ** 2).mean()
    reg_smooth = ((logQ[:, :, 1:] - logQ[:, :, :-1]) ** 2).mean()

    loss = task_loss + REG_Q_W * reg_q + REG_SMOOTH_W * reg_smooth
    metrics["loss"] = float(loss.item())
    return loss, metrics

# =========================
# 6. Train / eval loops
# =========================
def run_epoch(loader, optimizer=None, train=True, stage="train", epoch_idx=0):
    global global_step
    model.train() if train else model.eval()

    total = 0
    sum_loss = 0.0
    sum_sound_acc = 0.0
    sum_aoa_mae = 0.0
    sum_dist_acc = 0.0
    skipped = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)

            loss, metrics = compute_loss_active(batch) if Active else compute_loss_passive(batch)

            if not torch.isfinite(loss):
                skipped += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                loss.backward()

                # ✅ IMPORTANT: Clip gradients IMMEDIATELY after backward, before checking for bad grads
                # This helps prevent gradient explosion before it becomes NaN/Inf
                if Active and (fb_params is not None) and (backend_params is not None) and len(fb_params) > 0:
                    torch.nn.utils.clip_grad_norm_(fb_params, 0.2)  # Reduced from 0.3 for more stability
                    torch.nn.utils.clip_grad_norm_(backend_params, 3.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                writer.add_scalar(f"{stage}/loss_step", metrics["loss"], global_step)
                writer.add_scalar(f"{stage}/sound_acc_step", metrics["sound_acc"], global_step)
                writer.add_scalar(f"{stage}/aoa_mae_step", metrics["aoa_mae"], global_step)
                writer.add_scalar(f"{stage}/dist_acc_step", metrics["dist_acc"], global_step)

                total_gn, bad_all = log_grads_tensorboard(
                    model, global_step,
                    fb_params=fb_params if Active else None,
                    backend_params=backend_params if Active else None,
                    hist_every=HIST_EVERY,
                    max_param_log=MAX_PARAM_LOG
                )

                if global_step % PRINT_EVERY == 0:
                    print(
                        f"[step {global_step:06d}] "
                        f"loss={metrics['loss']:.4f} | "
                        f"sound_acc={metrics['sound_acc']:.3f} | "
                        f"aoa_mae={metrics['aoa_mae']:.3f} | "
                        f"dist_acc={metrics['dist_acc']:.3f} | "
                        f"grad_norm={total_gn:.4f} bad={bad_all}"
                    )

                # Check for bad grads AFTER clipping
                bad_grad = False
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        print(f"[BAD GRAD] {name}")
                        bad_grad = True
                        break

                if bad_grad:
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    global_step += 1
                    continue

                optimizer.step()
                global_step += 1

            bs = batch[0].shape[0]
            total += bs
            sum_loss      += metrics["loss"] * bs
            sum_sound_acc += metrics["sound_acc"] * bs
            sum_aoa_mae   += metrics["aoa_mae"] * bs
            sum_dist_acc  += metrics["dist_acc"] * bs

    if total == 0:
        out = {"loss": float("nan"), "sound_acc": 0.0, "aoa_mae": float("nan"),
               "dist_acc": 0.0, "skipped": skipped}
    else:
        out = {
            "loss": sum_loss / total,
            "sound_acc": sum_sound_acc / total,
            "aoa_mae": sum_aoa_mae / total,
            "dist_acc": sum_dist_acc / total,
            "skipped": skipped
        }

    writer.add_scalar(f"{stage}/loss_epoch", out["loss"], epoch_idx)
    writer.add_scalar(f"{stage}/sound_acc_epoch", out["sound_acc"], epoch_idx)
    writer.add_scalar(f"{stage}/aoa_mae_epoch", out["aoa_mae"], epoch_idx)
    writer.add_scalar(f"{stage}/dist_acc_epoch", out["dist_acc"], epoch_idx)
    writer.add_scalar(f"{stage}/skipped_epoch", out["skipped"], epoch_idx)

    return out

# =========================
# 7. Training
# =========================
best_val_tuple = None
best_val_loss = float("inf")
history = {"train": [], "val": []}

sanity_debug_one_batch(model, train_loader)

# =========================
# Optimizer
# =========================
if Active:
    # ✅ If frontend has NO trainable params, don't create an empty param group.
    param_groups = []
    if fb_params is not None and len(fb_params) > 0:
        param_groups.append({"params": fb_params, "lr": LR_FB})
    param_groups.append({"params": backend_params, "lr": LR_BACKEND})

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=WEIGHT_DECAY,
        eps=1e-7
    )
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_BACKEND, weight_decay=WEIGHT_DECAY, eps=1e-7)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

print("\n===== One-stage joint training (front-end + back-end) =====")
print("LR_FB:", LR_FB, "LR_BACKEND:", LR_BACKEND)
if Active:
    fb_numel = sum(p.numel() for p in fb_params) if fb_params is not None else 0
    be_numel = sum(p.numel() for p in backend_params) if backend_params is not None else 0
    print(f"[Fixed front-end Q?] {bool(FIXED_FRONTEND_Q)} (effective={bool(FIXED_FRONTEND_Q) and bool(Active)})")
    print(f"[Q-controller frozen only?] {bool(FREEZE_Q_CONTROLLER_ONLY)} (effective={bool(FREEZE_Q_CONTROLLER_ONLY) and bool(Active) and (not FIXED_FRONTEND_Q)})")
    print(f"[frontend trainable numel] {fb_numel}")
    print(f"[backend  trainable numel] {be_numel}")

for e in range(1, EPOCHS + 1):
    tr = run_epoch(train_loader, optimizer=optimizer, train=True,  stage="train", epoch_idx=e)
    va = run_epoch(val_loader,   optimizer=None,      train=False, stage="val",   epoch_idx=e)

    history["train"].append(tr)
    history["val"].append(va)

    print(f"[{e:03d}] "
          f"train_loss={tr['loss']:.4f} (skip={tr['skipped']}), "
          f"val_loss={va['loss']:.4f} (skip={va['skipped']}), "
          f"val_sound_acc={va['sound_acc']:.3f}, "
          f"val_aoa_mae={va['aoa_mae']:.3f}, "
          f"val_dist_acc={va['dist_acc']:.3f}")

    if torch.isfinite(torch.tensor(va["loss"])) and va["loss"] < best_val_loss:
        best_val_loss = va["loss"]
    scheduler.step(va["loss"])

    curr_tuple = (va["sound_acc"], va["aoa_mae"], va["dist_acc"])
    if torch.isfinite(torch.tensor(curr_tuple[0])) and torch.isfinite(torch.tensor(curr_tuple[1])) and torch.isfinite(torch.tensor(curr_tuple[2])):
        if _is_better_tuple(curr_tuple, best_val_tuple):
            best_val_tuple = curr_tuple
            torch.save(model.state_dict(), CKPT_BEST)
            print(f"  🔥 Saved new best model by metrics: sound_acc={curr_tuple[0]:.4f}, "
                  f"aoa_mae={curr_tuple[1]:.4f}, dist_acc={curr_tuple[2]:.4f}")
            print("     ->", CKPT_BEST)

    if SAVE_EVERY_EPOCH:
        ckpt_epoch = os.path.join(CKPT_DIR, f"epoch{e:03d}.pth")
        torch.save(model.state_dict(), ckpt_epoch)

print("Training finished.")

torch.save(model.state_dict(), CKPT_LAST)
with open(HIST_PATH, "w") as f:
    json.dump(history, f, indent=2)
print("[Saved] last:", CKPT_LAST)
print("[Saved] history:", HIST_PATH)

# =========================
# 8. Test + Q visualization
# =========================
print("\nEvaluating on test set...")
if os.path.exists(CKPT_BEST):
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device))
model.to(device)
model.eval()

te = run_epoch(test_loader, optimizer=None, train=False, stage="test", epoch_idx=0)
print("Test metrics:", te)

with open(os.path.join(JSON_DIR, "test_metrics.json"), "w") as f:
    json.dump(te, f, indent=2)

if Active:
    print("\nVisualizing Q...")
    visualize_Q_LR(
        model=model,
        dataloader=test_loader,
        device=device,
        save_dir=VIS_DIR,
        max_batches=5,
        sample_per_batch=1
    )

writer.flush()
writer.close()
print("[TensorBoard] closed.")
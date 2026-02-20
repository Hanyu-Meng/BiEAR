# evaluate_biear.py
# Evaluation for BIEAR/DeepEar: overall + per-speaker (1/2/3) metrics.
# All metrics use the same definition as train_torch.py: average over all N*8 sectors (no "active-only" filtering).
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_torch import build_model, build_model_active, build_model_active_single_controller, N_SECTORS, N_DIST_CLASS, DATA_DIM
from data import load_arrays_from_h5, DeepEarH5Dataset, DeepEarH5Dataset_Active

# =========================
# 1. 路径 & 配置
# =========================

# Option 1: Load from trained model's settings.json (recommended)
# Set this to the checkpoint path or the run directory
# CHECKPOINT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.45_0.35_run20260128-150702_dq1_lo0.3_hi5_rel_/checkpoints/best.pth"
# CHECKPOINT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.50_0.30_run20260116-042310_dq1_lo0.3_hi5_rel_/checkpoints/best.pth"
CHECKPOINT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-single_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.45_0.35_run20260216-020256_dq2_lo0.5_hi5_abs_single-ctrl/checkpoints/best.pth"
# CHECKPOINT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-single_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.45_0.35_run20260216-015951_dq1_lo0.3_hi5_rel_single-ctrl/checkpoints/best.pth"
# CHECKPOINT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.45_0.35_run20260128-150702_dq1_lo0.3_hi5_rel_/checkpoints/last.pth"
# CHECKPOINT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/biear_roommix_active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_roommix1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.45_0.35_run20260205-144509_dq1_lo0.3_hi5_rel_/checkpoints/best.pth"
# Or specify the run directory:
# RUN_DIR = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_room_tune/roomtune-spirit_active_cc-1_alpha0_tune0.10_tf1_tb1_run20260115-163903"
# Option 2: Manual configuration (if settings.json not available)
# Active = True   # True for active model (wav input), False for passive (feature input)
# USE_CC = True
# ALPHA = 0
# FIXED_FRONTEND_Q = False
# DELTAQ_BASE = 1.0
# DELTAQ_LOW_FACTOR = 0.3
# DELTAQ_HIGH_FACTOR = 5.0
# DELTAQ_MODE = "relative"
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Try to load configuration from checkpoint's settings.json
def load_settings_from_ckpt(ckpt_path):
    """Load model settings from checkpoint directory"""
    ckpt_dir = os.path.dirname(ckpt_path)
    settings_path = os.path.join(ckpt_dir, "..", "meta", "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            settings = json.load(f)
        return settings
    
    # Alternative: try to find settings.json in parent directories
    for parent in [ckpt_dir, os.path.dirname(ckpt_dir), os.path.dirname(os.path.dirname(ckpt_dir))]:
        alt_path = os.path.join(parent, "meta", "settings.json")
        if os.path.exists(alt_path):
            with open(alt_path, "r") as f:
                settings = json.load(f)
            return settings
    
    return None

# Load settings if checkpoint path is provided
if 'CHECKPOINT_PATH' in locals() and CHECKPOINT_PATH:
    settings = load_settings_from_ckpt(CHECKPOINT_PATH)
    if settings:
        print("[Config] Loading model settings from checkpoint...")
        Active = settings.get("Active", True)
        USE_CC = settings.get("USE_CC", True)
        ALPHA = settings.get("ALPHA", 0)
        FIXED_FRONTEND_Q = settings.get("FIXED_FRONTEND_Q", False)
        DELTAQ_BASE = settings.get("DELTAQ_BASE", 1.0)
        DELTAQ_LOW_FACTOR = settings.get("DELTAQ_LOW_FACTOR", 0.3)
        DELTAQ_HIGH_FACTOR = settings.get("DELTAQ_HIGH_FACTOR", 5.0)
        DELTAQ_MODE = settings.get("DELTAQ_MODE", "relative")
        Controller_Mode = settings.get("Controller_Mode", "dual")
        WEIGHT_PATH = CHECKPOINT_PATH
        print(f"  Active={Active}, USE_CC={USE_CC}, ALPHA={ALPHA}, FIXED_FRONTEND_Q={FIXED_FRONTEND_Q}")
        print(f"  DELTAQ_MODE={DELTAQ_MODE}, Controller_Mode={Controller_Mode}")
    else:
        print("[Config] No settings.json found. Please set manual configuration or provide correct checkpoint path.")
        raise ValueError("Cannot load model configuration. Please check CHECKPOINT_PATH or set manual config.")
else:
    # Manual configuration (fallback)
    print("[Config] Using manual configuration...")
    if 'WEIGHT_PATH' not in locals():
        WEIGHT_PATH = "deepear_torch_ILD_IPD_last.pth"
    if 'Active' not in locals():
        Active = True  # Default to passive
    if 'USE_CC' not in locals():
        USE_CC = True
    if 'ALPHA' not in locals():
        ALPHA = 0
    if 'FIXED_FRONTEND_Q' not in locals():
        FIXED_FRONTEND_Q = False
    if 'DELTAQ_BASE' not in locals():
        DELTAQ_BASE = 1.0
    if 'DELTAQ_LOW_FACTOR' not in locals():
        DELTAQ_LOW_FACTOR = 0.3
    if 'DELTAQ_HIGH_FACTOR' not in locals():
        DELTAQ_HIGH_FACTOR = 5.0
    if 'DELTAQ_MODE' not in locals():
        DELTAQ_MODE = "absolute"
    if 'Controller_Mode' not in locals():
        Controller_Mode = "dual"

# Test data path (modify these)
if Active:
    # For active model, use wav H5 files
    ROOT = "/media/mengh/SharedData/hanyu/DeepEar/anechoic_dataset"
    test_h5 = f"{ROOT}/anechoic_test2_active_wav.h5"
else:
    # For passive model, use feature H5 files
    ROOT = "/media/mengh/SharedData/hanyu/DeepEar/anechoic_dataset"
    test_h5 = f"{ROOT}/anechoic_test2_gt_group_phase.h5"


# =========================
# 2. 读取 test 数据
# =========================

print(f"\n[Data] Loading test set from {test_h5} ...")
if not os.path.exists(test_h5):
    raise FileNotFoundError(f"Test H5 file not found: {test_h5}")

if Active:
    # Active model: use DeepEarH5Dataset_Active (wav input)
    test_ds = DeepEarH5Dataset_Active(test_h5)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"  Active model: loaded {len(test_ds)} wav samples")
else:
    # Passive model: use DeepEarH5Dataset (feature input)
    test_ds = DeepEarH5Dataset(test_h5)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"  Passive model: loaded {len(test_ds)} feature samples")

N = len(test_ds)
print(f"Test samples: {N}")


# =========================
# 3. 加载模型 & 权重
# =========================

print(f"\n[Model] Building {'active' if Active else 'passive'} model (Controller_Mode={Controller_Mode})...")
if Active:
    if Controller_Mode == "single":
        model = build_model_active_single_controller(
            use_cc=USE_CC,
            fb_alpha=ALPHA,
            fixed_frontend_q=bool(FIXED_FRONTEND_Q),
            deltaQ_base=DELTAQ_BASE,
            deltaQ_low_factor=DELTAQ_LOW_FACTOR,
            deltaQ_high_factor=DELTAQ_HIGH_FACTOR,
            deltaQ_mode=DELTAQ_MODE,
        ).to(device)
    else:
        model = build_model_active(
            use_cc=USE_CC,
            fb_alpha=ALPHA,
            fixed_frontend_q=bool(FIXED_FRONTEND_Q),
            deltaQ_base=DELTAQ_BASE,
            deltaQ_low_factor=DELTAQ_LOW_FACTOR,
            deltaQ_high_factor=DELTAQ_HIGH_FACTOR,
            deltaQ_mode=DELTAQ_MODE,
        ).to(device)
else:
    model = build_model(use_cc=USE_CC).to(device)

# Load checkpoint (handle different formats)
def _load_state_dict_flexible(ckpt_obj):
    """Support different checkpoint formats"""
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        return ckpt_obj["model"]
    return ckpt_obj

print(f"[Model] Loading weights from {WEIGHT_PATH}...")
ckpt = torch.load(WEIGHT_PATH, map_location=device)
state = _load_state_dict_flexible(ckpt)
missing, unexpected = model.load_state_dict(state, strict=False)
model.eval()

print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
if len(missing) > 0 and len(missing) < 10:
    print(f"    Missing example: {missing[:3]}")
if len(unexpected) > 0 and len(unexpected) < 10:
    print(f"    Unexpected example: {unexpected[:3]}")
print("  Model loaded successfully!")


# =========================
# 4. 从 y_all 提取 GT: sound / aoa / distance
# =========================

def extract_gt_arrays(y_all):
    """
    y_all: shape (N, 56)  numpy array
    返回：
        sound_gt: (N, 8)       每扇区是否有声源（0/1 浮点）
        aoa_gt_norm: (N, 8)   归一化 AoA (0~1)
        aoa_gt_deg: (N, 8)    绝对角度（度），无源处为 NaN
        dist_gt: (N, 8)       距离类别索引 0~4
    """
    N = y_all.shape[0]
    sound_gt = np.zeros((N, N_SECTORS), dtype=np.float32)
    aoa_gt_norm = np.zeros((N, N_SECTORS), dtype=np.float32)
    aoa_gt_deg = np.full((N, N_SECTORS), np.nan, dtype=np.float32)
    dist_gt = np.zeros((N, N_SECTORS), dtype=np.int64)

    for k in range(N_SECTORS):
        base = 7 * k
        sound_gt[:, k] = y_all[:, base]
        aoa_gt_norm[:, k] = y_all[:, base + 1]
        # Handle both 4 and 5 distance classes
        if N_DIST_CLASS == 5:
            dist_slice = y_all[:, base + 2: base + 7]  # (N, 5)
        elif N_DIST_CLASS == 4:
            dist_slice = y_all[:, base + 2: base + 6]  # (N, 4)
        else:
            raise ValueError(f"Unexpected N_DIST_CLASS = {N_DIST_CLASS}")
        dist_gt[:, k] = np.argmax(dist_slice, axis=1)

        # 计算绝对角度
        sector_min = 45.0 * k
        mask = sound_gt[:, k] > 0.5
        aoa_gt_deg[mask, k] = sector_min + aoa_gt_norm[mask, k] * 45.0

    return sound_gt, aoa_gt_norm, aoa_gt_deg, dist_gt


# =========================
# 5. 从 logits 提取预测数组
# =========================

def extract_pred_arrays_from_logits(sound_logits, aoa_pred, dist_logits, sound_thresh=0.5):
    """
    sound_logits: (N, 8)  torch 或 numpy
    aoa_pred:     (N, 8)  归一化 AoA 预测 (0~1)
    dist_logits:  (N, 8, 5)

    返回：
        sound_pred_prob: (N, 8)   sound probability
        aoa_pred_norm:   (N, 8)
        aoa_pred_deg:    (N, 8)
        dist_pred_class: (N, 8)   0~4
    """
    if isinstance(sound_logits, torch.Tensor):
        sound_logits = sound_logits.cpu().numpy()
    if isinstance(aoa_pred, torch.Tensor):
        aoa_pred = aoa_pred.cpu().numpy()
    if isinstance(dist_logits, torch.Tensor):
        dist_logits = dist_logits.cpu().numpy()

    # sound prob
    sound_pred_prob = 1.0 / (1.0 + np.exp(-sound_logits))  # sigmoid

    aoa_pred_norm = aoa_pred.astype(np.float32)
    dist_pred_class = dist_logits.argmax(axis=-1).astype(np.int64)  # (N,8)

    N = sound_pred_prob.shape[0]
    aoa_pred_deg = np.full((N, N_SECTORS), np.nan, dtype=np.float32)

    for k in range(N_SECTORS):
        sector_min = 45.0 * k
        sound_vec = sound_pred_prob[:, k]
        aoa_vec = aoa_pred_norm[:, k]
        mask = sound_vec > sound_thresh
        aoa_pred_deg[mask, k] = sector_min + aoa_vec[mask] * 45.0

    return sound_pred_prob, aoa_pred_norm, aoa_pred_deg, dist_pred_class


# =========================
# 6. 先跑一遍 test，收集所有 logits 和 GT
# =========================

all_sound_logits = []
all_aoa_pred = []
all_dist_logits = []
all_y = []

print(f"\n[Evaluation] Running inference on test set...")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if Active:
            # Active model: (wavL, wavR, x3, y)
            wavL, wavR, x3, y = batch
            wavL = wavL.to(device).float()
            wavR = wavR.to(device).float()
            x3 = x3.to(device).float()
            y = y.to(device).float()
            
            # Normalize wav if needed
            maxabsL = wavL.abs().max().item()
            maxabsR = wavR.abs().max().item()
            if maxabsL > 2.0 or maxabsR > 2.0:
                wavL = wavL / 32768.0
                wavR = wavR / 32768.0
            wavL = torch.clamp(wavL, -1.0, 1.0)
            wavR = torch.clamp(wavR, -1.0, 1.0)
            
            # Sanitize x3
            x3 = x3.float()
            x3 = torch.nan_to_num(x3, nan=0.0, posinf=0.0, neginf=0.0)
            maxabs = x3.abs().amax(dim=1, keepdim=True)
            scale = torch.clamp(maxabs, min=1.0)
            x3 = x3 / scale
            x3 = torch.clamp(x3, -5.0, 5.0)
            
            sound_logits, aoa_pred, dist_logits = model(wavL, wavR, x3)
        else:
            # Passive model: (x1, x2, x3, x4, x5, y)
            x1, x2, x3, x4, x5, y = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            x5 = x5.to(device)
            y = y.to(device)
            
            sound_logits, aoa_pred, dist_logits = model(x1, x2, x3, x4, x5)
        
        all_sound_logits.append(sound_logits.cpu())
        all_aoa_pred.append(aoa_pred.cpu())
        all_dist_logits.append(dist_logits.cpu())
        all_y.append(y.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1} batches...")

sound_logits_all = torch.cat(all_sound_logits, dim=0)   # (N,8)
aoa_pred_all     = torch.cat(all_aoa_pred,   dim=0)     # (N,8)
dist_logits_all  = torch.cat(all_dist_logits,dim=0)     # (N,8,5)
y_all            = torch.cat(all_y,          dim=0)     # (N,56)

# 转成 numpy
sound_logits_all = sound_logits_all.numpy()
aoa_pred_all     = aoa_pred_all.numpy()
dist_logits_all  = dist_logits_all.numpy()
y_all_np         = y_all.numpy().astype(np.float32)

print("Collected predictions and GT from test set.")


# =========================
# 7. 抽取 GT & Pred 数组
# =========================

sound_gt, aoa_gt_norm, aoa_gt_deg, dist_gt = extract_gt_arrays(y_all_np)
sound_pred, aoa_pred_norm, aoa_pred_deg, dist_pred = extract_pred_arrays_from_logits(
    sound_logits_all, aoa_pred_all, dist_logits_all, sound_thresh=0.5
)

print("sound_gt shape      :", sound_gt.shape)
print("aoa_gt_deg shape    :", aoa_gt_deg.shape)
print("dist_gt shape       :", dist_gt.shape)
print("sound_pred shape    :", sound_pred.shape)
print("aoa_pred_deg shape  :", aoa_pred_deg.shape)
print("dist_pred shape     :", dist_pred.shape)


# =========================
# 8. 与训练一致的指标计算（所有 sector 平均，与 train_torch.py 一致）
# =========================

def compute_metrics_like_train(sound_gt, sound_pred, aoa_gt_norm, aoa_pred_norm, dist_gt, dist_pred):
    """
    与 train_torch.py 完全一致：在给定数组上对 **所有 sector** 计算三项指标。
    - sound_acc: mean((pred>0.5)==(gt>0.5))
    - aoa_mae:   mean(|aoa_pred_norm - aoa_gt_norm|)  归一化空间
    - dist_acc:  mean(dist_pred == dist_gt)
    输入形状均为 (M, 8)。
    """
    gt_has = sound_gt > 0.5
    pred_has = sound_pred > 0.5
    sound_acc = np.mean(gt_has == pred_has)
    aoa_mae = np.mean(np.abs(aoa_pred_norm.astype(np.float64) - aoa_gt_norm.astype(np.float64)))
    dist_acc = np.mean(dist_pred.flatten() == dist_gt.flatten())
    return {"sound_acc": float(sound_acc), "aoa_mae": float(aoa_mae), "dist_acc": float(dist_acc)}


def analyze_for_n_sources(n_src,
                          sound_gt, sound_pred,
                          aoa_gt_norm, aoa_pred_norm,
                          dist_gt, dist_pred):
    """
    在 GT 中声源个数 == n_src 的样本上，用与训练 **完全一致** 的定义计算三项指标：
    对所有 M×8 个 sector 取平均（不限于「GT 与预测都有源」的 sector）。
    """

    num_sources = (sound_gt > 0.5).sum(axis=1)  # (N,)
    sample_mask = (num_sources == n_src)

    if sample_mask.sum() == 0:
        print(f"\n### [n_src = {n_src}] No samples found, skip.")
        return None

    # 子集：(M, 8)，与整体/训练一样对全部 sector 算指标
    sg = sound_gt[sample_mask]
    sp = sound_pred[sample_mask]
    ag_norm = aoa_gt_norm[sample_mask]
    ap_norm = aoa_pred_norm[sample_mask]
    dg = dist_gt[sample_mask]
    dp = dist_pred[sample_mask]

    metrics = compute_metrics_like_train(sg, sp, ag_norm, ap_norm, dg, dp)
    n_samples = int(sample_mask.sum())

    print(f"\n==================== [n_src = {n_src}] ====================")
    print(f"#samples with {n_src} active sources : {n_samples}")
    print(f"  sound_acc (all sectors): {metrics['sound_acc']*100:.2f}%")
    print(f"  aoa_mae   (all sectors): {metrics['aoa_mae']:.4f}")
    print(f"  dist_acc  (all sectors): {metrics['dist_acc']*100:.2f}%")
    return metrics


# =========================
# 9. 整体评估（与 train_torch / test_metrics.json 完全一致：所有 sector 平均）
# =========================
print("\n" + "="*60)
print("OVERALL EVALUATION (All Samples, all sectors — same as training)")
print("="*60)
print(f"Total test samples: {len(sound_gt)}")

metrics_overall = compute_metrics_like_train(
    sound_gt, sound_pred, aoa_gt_norm, aoa_pred_norm, dist_gt, dist_pred
)
print(f"  sound_acc : {metrics_overall['sound_acc']*100:.2f}%")
print(f"  aoa_mae   : {metrics_overall['aoa_mae']:.4f}")
print(f"  dist_acc  : {metrics_overall['dist_acc']*100:.2f}%")

# 统计不同 n_src 的样本分布
num_sources = (sound_gt > 0.5).sum(axis=1)
for n_src in [1, 2, 3]:
    count = (num_sources == n_src).sum()
    print(f"Samples with {n_src} source(s): {count} ({count/len(sound_gt)*100:.1f}%)")
print("="*60)

# =========================
# 10. 分别评估 1 / 2 / 3 speakers（与训练一致：该子集内所有 sector 平均）
# =========================

results_by_speaker = {}
for n_src in [1, 2, 3]:
    m = analyze_for_n_sources(
        n_src,
        sound_gt=sound_gt,
        sound_pred=sound_pred,
        aoa_gt_norm=aoa_gt_norm,
        aoa_pred_norm=aoa_pred_norm,
        dist_gt=dist_gt,
        dist_pred=dist_pred,
    )
    if m is not None:
        results_by_speaker[f"{n_src}spk"] = m

# Optional: save overall + per-speaker metrics (same format as test_metrics.json) for comparison
if results_by_speaker or metrics_overall:
    out = {"overall": metrics_overall, **results_by_speaker}
    ckpt_dir = os.path.dirname(WEIGHT_PATH) if WEIGHT_PATH else "."
    out_path = os.path.abspath(os.path.join(ckpt_dir, "..", "evaluate_biear_metrics.json"))
    try:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved metrics to {out_path}")
    except Exception as e:
        print(f"\nCould not save metrics: {e}")

print("\nDone.")
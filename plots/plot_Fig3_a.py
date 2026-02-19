"""
Plot Q value evolution over time for three frequency bands (low/mid/high).

This script visualizes how Q values change over time frames for three
representative frequency bands with enhanced, beautiful visualization.
"""
#%%
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
# 远程/无显示器：用 Agg 不弹窗，图会保存到文件并打印路径；有 DISPLAY 则尝试弹窗
if not os.environ.get("DISPLAY", "").strip():
    matplotlib.use("Agg")
    _SAVE_PLOT_TO_FILE = True
else:
    _SAVE_PLOT_TO_FILE = False
    for _backend in ("TkAgg", "Qt5Agg", "GTK4Agg", "WXAgg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

from model_torch import build_model_active
from data import DeepEarH5Dataset_Active

# =========================
# USER CONFIG
# =========================
CKPT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb5e-05_lrbe0.0001_wd1e-05_lossw0.20_0.45_0.35_run20260128-150702_dq1_lo0.3_hi5_rel_/checkpoints/best.pth"
# TEST_H5 = "/media/mengh/SharedData/hanyu/DeepEar/auditorium3_dataset/auditorium3_test1_active_wav.h5"
TEST_H5 = "/media/mengh/SharedData/hanyu/DeepEar/anechoic_dataset/anechoic_test1_active_wav.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_CC = True
FB_ALPHA = 0
FIXED_FRONTEND_Q = False
DELTAQ_BASE = 1.0
DELTAQ_LOW_FACTOR = 0.3
DELTAQ_HIGH_FACTOR = 5.0
DELTAQ_MODE = "relative"

BATCH_SIZE = 1
N_SAMPLES_TO_PLOT = 20
SAMPLE_IDX = 12  # 只画这一个样本；设为 None 才会按 N_SAMPLES_TO_PLOT 画多张

LOW_IDX_FRAC, MID_IDX_FRAC, HIGH_IDX_FRAC = 0.10, 0.40, 0.80
FIXED_BAND_IDXS = None

# Plot style (align with plot_Fig3_b.py, flatter)
FIG_SIZE_SUBPLOTS = (6, 4)
FIG_SIZE_OVERLAY = (5, 3.5)
FIG_SIZE_LR = (6, 2.5)   # flatter: same as plot_Fig3_b.py
DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 10
LEGEND_SIZE = 10

# Color scheme: 红 / 蓝 / 黄，柔和易辨，solid + dashed 略浅
COLORS = {
    'low': '#c82423',
    'low_light': '#e08a86',
    'mid': '#2878b5',
    'mid_light': '#5b8cbc',
    'high': '#FFAA00', 
    'high_light': '#d4b84a',
    'q0_line': '#7d8a96',
}

LINE_WIDTH_QT = 2.0
LINE_WIDTH_Q0 = 3.0
LINE_STYLE_Q0 = '--'
ALPHA_QT = 0.95
ALPHA_Q0 = 0.88

GRID_ALPHA = 0.4
GRID_STYLE = '--'
GRID_WIDTH = 1.4
SPINE_WIDTH = 1.2
X_LIM = (-0.5, 18.5)

PLOT_SUBPLOTS = False
PLOT_OVERLAY = False
PLOT_LR_TOGETHER = True

SHOW_LEGEND = False
SHOW_STATS_BOX = False

# 远程登录时设为 True：不弹窗，把图保存到脚本所在目录（q_evolution_sample_*.png）并打印路径，
# 用 scp 下载或 VS Code Remote 直接打开该 PNG 即可看图
REMOTE_SAVE = True

# =========================
# Helpers
# =========================
@torch.no_grad()
def _sanitize_x3(x3: torch.Tensor) -> torch.Tensor:
    x3 = torch.nan_to_num(x3.float(), nan=0.0, posinf=0.0, neginf=0.0)
    maxabs = x3.abs().amax(dim=1, keepdim=True)
    x3 = x3 / torch.clamp(maxabs, min=1.0)
    return torch.clamp(x3, -5.0, 5.0)


def _load_state_dict_flexible(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        return ckpt_obj["model"]
    return ckpt_obj


def pick_band_indices(Nbands: int):
    if FIXED_BAND_IDXS is not None:
        idxs = list(FIXED_BAND_IDXS)
    else:
        idxs = [
            int(round((Nbands - 1) * LOW_IDX_FRAC)),
            int(round((Nbands - 1) * MID_IDX_FRAC)),
            int(round((Nbands - 1) * HIGH_IDX_FRAC)),
        ]
    idxs = [max(0, min(Nbands - 1, i)) for i in idxs]
    uniq = []
    for i in idxs:
        if i not in uniq:
            uniq.append(i)
    while len(uniq) < 3:
        cand = min(Nbands - 1, uniq[-1] + 1)
        if cand not in uniq:
            uniq.append(cand)
        else:
            break
    return uniq[:3]


# =========================
# Plot functions
# =========================
def plot_q_evolution_subplots(Qt, Q0, fcs, band_idxs, band_labels, colors_dict, ear_name, sample_idx, fb_alpha):
    T = Qt.shape[0]
    t_axis = np.arange(T)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE, 'axes.titlesize': TITLE_SIZE, 'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE - 1, 'ytick.labelsize': LABEL_SIZE - 1,
        'legend.fontsize': LEGEND_SIZE, 'figure.titlesize': TITLE_SIZE + 2,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE_SUBPLOTS, dpi=DPI, sharex=True, facecolor='white')
    color_map = {'Low': colors_dict['low'], 'Mid': colors_dict['mid'], 'High': colors_dict['high']}
    color_map_q0 = {
        'Low': colors_dict.get('low_light', colors_dict['q0_line']),
        'Mid': colors_dict.get('mid_light', colors_dict['q0_line']),
        'High': colors_dict.get('high_light', colors_dict['q0_line']),
    }
    for ax, name, bi in zip(axes, band_labels, band_idxs):
        q_line = Qt[:, bi].numpy()
        q0 = float(Q0[bi])
        fc_hz = float(fcs[bi])
        color = color_map.get(name, colors_dict['low'])
        color_q0 = color_map_q0.get(name, colors_dict['q0_line'])
        ax.plot(t_axis, q_line, color=color, linewidth=LINE_WIDTH_QT, label="Q(t)", alpha=ALPHA_QT, zorder=3)
        ax.axhline(y=q0, color=color_q0, linestyle=LINE_STYLE_Q0, linewidth=LINE_WIDTH_Q0, label="Q0", alpha=ALPHA_Q0, zorder=2)
        ax.set_ylabel("Q Value", fontweight='bold')
        ax.set_xlim(X_LIM)
        y_min = min(q_line.min(), q0) * 0.95
        y_max = max(q_line.max(), q0) * 1.05
        ax.set_ylim([y_min, y_max])
        ax.set_title(f"{ear_name} Ear | {name} Frequency Band (fc = {fc_hz:.0f} Hz)", fontweight='bold', pad=10)
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_WIDTH)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
        if SHOW_LEGEND:
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, edgecolor='gray', facecolor='white')
        ax.text(0.98, 0.95, f"Q0 = {q0:.3f}", transform=ax.transAxes, fontsize=LEGEND_SIZE - 1, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7), family='serif')
    axes[-1].set_xlabel("Frame Index t", fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_q_evolution_overlay(Qt, Q0, fcs, band_idxs, band_labels, colors_dict, ear_name, sample_idx, fb_alpha):
    T = Qt.shape[0]
    t_axis = np.arange(T)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE, 'axes.titlesize': TITLE_SIZE, 'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE - 1, 'ytick.labelsize': LABEL_SIZE - 1,
        'legend.fontsize': LEGEND_SIZE, 'figure.titlesize': TITLE_SIZE + 2,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })
    fig, ax = plt.subplots(figsize=FIG_SIZE_OVERLAY, dpi=DPI, facecolor='white')
    color_map = {'Low': colors_dict['low'], 'Mid': colors_dict['mid'], 'High': colors_dict['high']}
    color_map_q0 = {
        'Low': colors_dict.get('low_light', colors_dict['q0_line']),
        'Mid': colors_dict.get('mid_light', colors_dict['q0_line']),
        'High': colors_dict.get('high_light', colors_dict['q0_line']),
    }
    for name, bi in zip(band_labels, band_idxs):
        fc_hz = float(fcs[bi])
        q_line = Qt[:, bi].numpy()
        color = color_map.get(name, colors_dict['low'])
        ax.plot(t_axis, q_line, color=color, linewidth=LINE_WIDTH_QT,
                label=f"{name} Band Q(t) (fc={fc_hz:.0f} Hz)", alpha=ALPHA_QT, zorder=3)
    q0_values = []
    for name, bi in zip(band_labels, band_idxs):
        fc_hz = float(fcs[bi])
        q0 = float(Q0[bi])
        q0_values.append(q0)
        color_q0 = color_map_q0.get(name, colors_dict['q0_line'])
        ax.axhline(y=q0, color=color_q0, linestyle=LINE_STYLE_Q0, linewidth=LINE_WIDTH_Q0, alpha=ALPHA_Q0, zorder=2)
        ax.text(t_axis[-1] + 0.3, q0, f"Q0 {name}\n({fc_hz:.0f}Hz)", va="center", fontsize=LEGEND_SIZE - 1, color=color_q0,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color_q0, linewidth=1))
    ax.set_xlim(X_LIM)
    y_min = min([Qt[:, bi].min() for bi in band_idxs] + q0_values) * 0.95
    y_max = max([Qt[:, bi].max() for bi in band_idxs] + q0_values) * 1.05
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("Frame Index t", fontweight='bold')
    ax.set_ylabel("Q Value", fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_WIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    if SHOW_LEGEND:
        ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, edgecolor='gray', facecolor='white', ncol=1)
    if SHOW_STATS_BOX:
        stats_text = []
        for name, bi in zip(band_labels, band_idxs):
            q_line = Qt[:, bi].numpy()
            q0 = float(Q0[bi])
            q_mean = float(q_line.mean())
            q_std = float(q_line.std())
            q_change = q_mean - q0
            stats_text.append(f"{name}: Q0={q0:.3f}, Q̄={q_mean:.3f}±{q_std:.3f}, Δ={q_change:+.3f}")
        stats_str = "\n".join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, fontsize=LEGEND_SIZE - 2, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    plt.tight_layout()
    return fig


def plot_q_evolution_lr_together(QtL, QtR, Q0, fcs, band_idxs, band_labels, colors_dict, sample_idx, fb_alpha):
    T = QtL.shape[0]
    t_axis = np.arange(T)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE, 'axes.titlesize': TITLE_SIZE, 'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE - 1, 'ytick.labelsize': LABEL_SIZE - 1,
        'legend.fontsize': LEGEND_SIZE, 'figure.titlesize': TITLE_SIZE + 2,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_LR, dpi=DPI, facecolor='white', sharey=True)
    color_map = {'Low': colors_dict['low'], 'Mid': colors_dict['mid'], 'High': colors_dict['high']}
    color_map_q0 = {
        'Low': colors_dict.get('low_light', colors_dict['q0_line']),
        'Mid': colors_dict.get('mid_light', colors_dict['q0_line']),
        'High': colors_dict.get('high_light', colors_dict['q0_line']),
    }
    for ax, Qt, ear_name in zip(axes, [QtL, QtR], ['Left', 'Right']):
        for name, bi in zip(band_labels, band_idxs):
            fc_hz = float(fcs[bi])
            q_line = Qt[:, bi].numpy()
            color = color_map.get(name, colors_dict['low'])
            ax.plot(t_axis, q_line, color=color, linewidth=LINE_WIDTH_QT,
                   label=f"{name} Band (fc={fc_hz:.0f} Hz)", alpha=ALPHA_QT, zorder=3)
        q0_values = []
        for name, bi in zip(band_labels, band_idxs):
            q0 = float(Q0[bi])
            q0_values.append(q0)
            color_q0 = color_map_q0.get(name, colors_dict['q0_line'])
            ax.axhline(y=q0, color=color_q0, linestyle=LINE_STYLE_Q0, linewidth=LINE_WIDTH_Q0, alpha=ALPHA_Q0, zorder=2)
        ax.set_xlim(X_LIM)
        y_min = min([Qt[:, bi].min() for bi in band_idxs] + q0_values) * 0.95
        y_max = max([Qt[:, bi].max() for bi in band_idxs] + q0_values) * 1.05
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Frame Index t", fontweight='bold')
        ax.set_title(f"{ear_name} Ear", fontweight='bold', pad=10, fontsize=TITLE_SIZE)
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_WIDTH)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
        if ear_name == 'Right' and SHOW_LEGEND:
            ax.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, edgecolor='gray', facecolor='white', ncol=1)
        if SHOW_STATS_BOX:
            stats_text = []
            for name, bi in zip(band_labels, band_idxs):
                q_line = Qt[:, bi].numpy()
                q0 = float(Q0[bi])
                q_mean = float(q_line.mean())
                q_change = q_mean - q0
                stats_text.append(f"{name}: Q0={q0:.3f}, Q̄={q_mean:.3f}, Δ={q_change:+.3f}")
            stats_str = "\n".join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, fontsize=LEGEND_SIZE - 2, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='serif')
    axes[0].set_ylabel("Q Value", fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    return fig


def plot_q_evolution_subplots_lr_together(QtL, QtR, Q0, fcs, band_idxs, band_labels, colors_dict, sample_idx, fb_alpha):
    T = QtL.shape[0]
    t_axis = np.arange(T)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE, 'axes.titlesize': TITLE_SIZE, 'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE - 1, 'ytick.labelsize': LABEL_SIZE - 1,
        'legend.fontsize': LEGEND_SIZE, 'figure.titlesize': TITLE_SIZE + 2,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=DPI, sharex='col', sharey='row', facecolor='white')
    color_map = {'Low': colors_dict['low'], 'Mid': colors_dict['mid'], 'High': colors_dict['high']}
    color_map_q0 = {
        'Low': colors_dict.get('low_light', colors_dict['q0_line']),
        'Mid': colors_dict.get('mid_light', colors_dict['q0_line']),
        'High': colors_dict.get('high_light', colors_dict['q0_line']),
    }
    for row_idx, (name, bi) in enumerate(zip(band_labels, band_idxs)):
        fc_hz = float(fcs[bi])
        q0 = float(Q0[bi])
        color = color_map.get(name, colors_dict['low'])
        color_q0 = color_map_q0.get(name, colors_dict['q0_line'])
        ax_left = axes[row_idx, 0]
        q_line_L = QtL[:, bi].numpy()
        ax_left.plot(t_axis, q_line_L, color=color, linewidth=LINE_WIDTH_QT, label="Q(t)", alpha=ALPHA_QT, zorder=3)
        ax_left.axhline(y=q0, color=color_q0, linestyle=LINE_STYLE_Q0, linewidth=LINE_WIDTH_Q0, label="Q0", alpha=ALPHA_Q0, zorder=2)
        ax_left.set_ylabel("Q Value", fontweight='bold')
        ax_left.set_xlim(X_LIM)
        ax_left.set_ylim([min(q_line_L.min(), q0) * 0.95, max(q_line_L.max(), q0) * 1.05])
        ax_left.set_title(f"Left Ear | {name} Band (fc = {fc_hz:.0f} Hz)", fontweight='bold', pad=10)
        ax_left.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_WIDTH)
        ax_left.set_axisbelow(True)
        if SHOW_LEGEND:
            ax_left.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, edgecolor='gray', facecolor='white')
        ax_left.text(0.98, 0.95, f"Q0 = {q0:.3f}", transform=ax_left.transAxes, fontsize=LEGEND_SIZE - 1, verticalalignment='top',
                     horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7), family='serif')
        ax_right = axes[row_idx, 1]
        q_line_R = QtR[:, bi].numpy()
        ax_right.plot(t_axis, q_line_R, color=color, linewidth=LINE_WIDTH_QT, label="Q(t)", alpha=ALPHA_QT, zorder=3)
        ax_right.axhline(y=q0, color=color_q0, linestyle=LINE_STYLE_Q0, linewidth=LINE_WIDTH_Q0, label="Q0", alpha=ALPHA_Q0, zorder=2)
        ax_right.set_xlim(X_LIM)
        ax_right.set_ylim([min(q_line_R.min(), q0) * 0.95, max(q_line_R.max(), q0) * 1.05])
        ax_right.set_title(f"Right Ear | {name} Band (fc = {fc_hz:.0f} Hz)", fontweight='bold', pad=10)
        ax_right.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_WIDTH)
        ax_right.set_axisbelow(True)
        if SHOW_LEGEND:
            ax_right.legend(loc='best', framealpha=0.9, fancybox=True, shadow=True, edgecolor='gray', facecolor='white')
        ax_right.text(0.98, 0.95, f"Q0 = {q0:.3f}", transform=ax_right.transAxes, fontsize=LEGEND_SIZE - 1, verticalalignment='top',
                      horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7), family='serif')
    axes[-1, 0].set_xlabel("Frame Index t", fontweight='bold')
    axes[-1, 1].set_xlabel("Frame Index t", fontweight='bold')
    fixed_q_str = "Fixed" if FIXED_FRONTEND_Q else "Adaptive"
    fig.suptitle(f"Q(t) Evolution Over Time: Left vs Right Ear (3 Bands)\nSample {sample_idx} | α={fb_alpha} | {fixed_q_str} Q",
                 fontweight='bold', y=0.995, fontsize=TITLE_SIZE + 2)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# =========================
# Main
# =========================
def main():
    assert os.path.exists(CKPT_PATH), f"CKPT not found: {CKPT_PATH}"
    assert os.path.exists(TEST_H5), f"Test H5 not found: {TEST_H5}"
    print(f"[Config] Device: {DEVICE}")
    if _SAVE_PLOT_TO_FILE or REMOTE_SAVE:
        print("[Config] Remote/save mode: figure will be saved to PNG (no window)")
    print(f"[Config] Plot subplots: {PLOT_SUBPLOTS}, overlay: {PLOT_OVERLAY}, LR together: {PLOT_LR_TOGETHER}")
    if SAMPLE_IDX is not None:
        print(f"[Config] Plotting only sample index: {SAMPLE_IDX} (single figure)")
    else:
        print(f"[Config] Plotting samples: 0 to {N_SAMPLES_TO_PLOT - 1} (multiple figures)")

    model = build_model_active(
        use_cc=USE_CC, fb_alpha=FB_ALPHA, fixed_frontend_q=bool(FIXED_FRONTEND_Q),
        deltaQ_base=DELTAQ_BASE, deltaQ_low_factor=DELTAQ_LOW_FACTOR,
        deltaQ_high_factor=DELTAQ_HIGH_FACTOR, deltaQ_mode=DELTAQ_MODE,
    ).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = _load_state_dict_flexible(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] Loaded: {CKPT_PATH}; missing: {len(missing)}, unexpected: {len(unexpected)}")
    model.eval()

    ds = DeepEarH5Dataset_Active(TEST_H5)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    fcs = model.bifb.fc.detach().cpu().numpy()
    Q0 = model.bifb.Q0.detach().cpu()
    Nbands = model.bifb.Nbands
    band_idxs = pick_band_indices(Nbands)
    band_idxs = sorted(band_idxs, key=lambda i: fcs[i])
    band_labels = ["Low", "Mid", "High"]
    print(f"[Model] Bands: {Nbands}; selected: {[(band_labels[j], band_idxs[j], fcs[band_idxs[j]]) for j in range(3)]}")

    n_plotted = 0
    for batch_idx, batch in enumerate(loader):
        if SAMPLE_IDX is not None and batch_idx != SAMPLE_IDX:
            n_plotted += 1
            continue
        wavL, wavR, x3, y = batch
        wavL = wavL.to(DEVICE).float()
        wavR = wavR.to(DEVICE).float()
        x3 = _sanitize_x3(x3.to(DEVICE))
        with torch.no_grad():
            YL, YR, QL, QR, XL, XR = model.bifb(wavL, wavR)
        QtL = QL[0].detach().cpu()
        QtR = QR[0].detach().cpu()
        current_sample_idx = SAMPLE_IDX if SAMPLE_IDX is not None else n_plotted
        print(f"\n[Sample {current_sample_idx}] Processing...")

        if PLOT_LR_TOGETHER:
            plot_q_evolution_lr_together(
                QtL=QtL, QtR=QtR, Q0=Q0, fcs=fcs, band_idxs=band_idxs,
                band_labels=band_labels, colors_dict=COLORS,
                sample_idx=current_sample_idx, fb_alpha=FB_ALPHA
            )
            # plot_q_evolution_subplots_lr_together(...)  # 3x2 grid, commented out

        if PLOT_SUBPLOTS:
            plot_q_evolution_subplots(Qt=QtL, Q0=Q0, fcs=fcs, band_idxs=band_idxs, band_labels=band_labels,
                                      colors_dict=COLORS, ear_name="Left", sample_idx=current_sample_idx, fb_alpha=FB_ALPHA)
            plot_q_evolution_subplots(Qt=QtR, Q0=Q0, fcs=fcs, band_idxs=band_idxs, band_labels=band_labels,
                                      colors_dict=COLORS, ear_name="Right", sample_idx=current_sample_idx, fb_alpha=FB_ALPHA)
        if PLOT_OVERLAY:
            plot_q_evolution_overlay(Qt=QtL, Q0=Q0, fcs=fcs, band_idxs=band_idxs, band_labels=band_labels,
                                     colors_dict=COLORS, ear_name="Left", sample_idx=current_sample_idx, fb_alpha=FB_ALPHA)
            plot_q_evolution_overlay(Qt=QtR, Q0=Q0, fcs=fcs, band_idxs=band_idxs, band_labels=band_labels,
                                     colors_dict=COLORS, ear_name="Right", sample_idx=current_sample_idx, fb_alpha=FB_ALPHA)

        if _SAVE_PLOT_TO_FILE or REMOTE_SAVE:
            out_dir = os.path.dirname(os.path.abspath(__file__))
            out_name = f"q_evolution_sample_{current_sample_idx}.png"
            out_path = os.path.join(out_dir, out_name)
            plt.gcf().savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.01, transparent=True)
            print(f"[Saved] {out_path}")
        else:
            plt.show()
        if SAMPLE_IDX is not None:
            break
        n_plotted += 1
        if n_plotted >= N_SAMPLES_TO_PLOT:
            break
    print("\n[Done] Visualization complete!")


if __name__ == "__main__":
    main()

# %%

"""
Plot Low/Mid/High three-band filter responses: fixed Q0 vs adaptive Q(t) at frame t=5.

One figure, two subplots (Left Ear | Right Ear). Each subplot shows 3 bands;
for each band: dashed = fixed filter (Q0), solid = adaptive filter (Q(t)).
Style (colors, font, size) aligned with plot_Fig3_v1.py.
"""
#%%
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
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
# USER CONFIG (align with plot_Fig3_v1)
# =========================
CKPT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb0.0001_lrbe0.0001_wd1e-05_lossw0.20_0.40_0.40_run20260114-143653_dq1_lo0.3_hi5_rel_absolute-deltaq/checkpoints/best.pth"
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
N_SAMPLES_TO_PLOT = 1
SAMPLE_IDX = 12 # only this sample; None = first N_SAMPLES_TO_PLOT

# Band selection (low / mid / high)
LOW_IDX_FRAC, MID_IDX_FRAC, HIGH_IDX_FRAC = 0.10, 0.40, 0.80
FIXED_BAND_IDXS = None

# Frame for filter comparison: t = 5
FRAME_TO_SHOW = 5

# Plot style (flatter)
FIG_SIZE = (6, 2.5)
DPI = 300

FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 10
LEGEND_SIZE = 10
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
USE_LOG_FREQ = True
SHOW_LEGEND = False

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


def build_W_from_Q(model, Q_bn: torch.Tensor, device):
    """Filter weight W from Q. Q_bn: (1, N) -> W: (1, N, F)."""
    f = model.bifb.f_fft.view(1, 1, -1).to(device)
    fc = model.bifb.fc.view(1, model.bifb.Nbands, 1).to(device)
    bw = (model.bifb.fc.unsqueeze(0).to(device) / (Q_bn + 1e-8)).unsqueeze(-1)
    W = torch.exp(-0.5 * ((f - fc) / (bw + 1e-12)) ** 2)
    W = torch.clamp(W, min=1e-12)
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-12)
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return W


# =========================
# Plot: one figure, two subplots (Left | Right), 3 bands each, fixed vs adaptive
# =========================
def plot_three_bands_filters_lr_together(
    model, Q0, QtL, QtR, fcs, frame_idx, band_idxs, band_labels, colors_dict, sample_idx
):
    """
    One figure, two subplots (Left Ear, Right Ear). Each subplot: 3 bands (Low/Mid/High),
    each band has fixed filter (Q0, dashed) and adaptive filter (Q(t), solid).
    """
    device = next(model.parameters()).device
    f = model.bifb.f_fft.detach().cpu().numpy()  # (F,)

    T = QtL.shape[0]
    t_idx = max(0, min(frame_idx, T - 1))
    QtL_frame = QtL[t_idx].view(1, -1).to(device)
    QtR_frame = QtR[t_idx].view(1, -1).to(device)
    Q0_1 = Q0.view(1, -1).to(device)

    with torch.no_grad():
        W0 = build_W_from_Q(model, Q0_1, device)[0].detach().cpu().numpy()   # (N, F)
        WL_t = build_W_from_Q(model, QtL_frame, device)[0].detach().cpu().numpy()  # (N, F)
        WR_t = build_W_from_Q(model, QtR_frame, device)[0].detach().cpu().numpy()  # (N, F)

    color_map = {'Low': colors_dict['low'], 'Mid': colors_dict['mid'], 'High': colors_dict['high']}
    color_map_q0 = {
        'Low': colors_dict.get('low_light', colors_dict['q0_line']),
        'Mid': colors_dict.get('mid_light', colors_dict['q0_line']),
        'High': colors_dict.get('high_light', colors_dict['q0_line']),
    }

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE, 'axes.titlesize': TITLE_SIZE, 'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE - 1, 'ytick.labelsize': LABEL_SIZE - 1,
        'legend.fontsize': LEGEND_SIZE, 'figure.titlesize': TITLE_SIZE + 2,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI, facecolor='white', sharey=True)

    for ax, Wt, ear_name in zip(axes, [WL_t, WR_t], ['Left', 'Right']):
        for name, bi in zip(band_labels, band_idxs):
            fc_hz = float(fcs[bi])
            c = color_map.get(name, colors_dict['low'])
            c_q0 = color_map_q0.get(name, colors_dict['q0_line'])
            ax.plot(f, Wt[bi], color=c, linestyle='-', linewidth=LINE_WIDTH_QT,
                    alpha=ALPHA_QT, zorder=2, label=f"{name} adaptive (t={frame_idx})" if ear_name == 'Right' else None)
            ax.plot(f, W0[bi], color=c_q0, linestyle=LINE_STYLE_Q0, linewidth=LINE_WIDTH_Q0,
                    alpha=ALPHA_Q0, zorder=3, label=f"{name} fixed" if ear_name == 'Right' else None)
        ax.set_xlabel("Frequency (Hz)", fontweight='bold')
        ax.set_title(f"{ear_name} Ear (t={frame_idx})", fontweight='bold', pad=10, fontsize=TITLE_SIZE)
        if ear_name == 'Left':
            ax.set_ylabel("Filter gain", fontweight='bold')
        fc_lo = float(fcs[band_idxs[0]])
        fc_hi = float(fcs[band_idxs[-1]])
        if USE_LOG_FREQ:
            ax.set_xscale('log')
            f_pos = f[f > 0]
            f_min_data = f_pos.min() if len(f_pos) else 1
            f_max_data = f.max()
            x_min = max(f_min_data, fc_lo * 0.25)
            x_max = min(f_max_data, fc_hi * 2.5)
            ax.set_xlim([x_min, x_max])
        else:
            margin = (fc_hi - fc_lo) * 0.5
            x_min = max(f.min(), fc_lo - margin)
            x_max = min(f.max(), fc_hi + margin)
            ax.set_xlim([x_min, x_max])
        y_max = max(W0[band_idxs].max(), Wt[band_idxs].max()) * 1.12
        ax.set_ylim([-0.01, y_max*1.2])
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_STYLE, linewidth=GRID_WIDTH)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
        if ear_name == 'Right' and SHOW_LEGEND:
            ax.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=LEGEND_SIZE - 1)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    return fig


# =========================
# Main
# =========================
def main():
    assert os.path.exists(CKPT_PATH), f"CKPT not found: {CKPT_PATH}"
    assert os.path.exists(TEST_H5), f"Test H5 not found: {TEST_H5}"

    print(f"[Config] Device: {DEVICE}")
    if _SAVE_PLOT_TO_FILE or REMOTE_SAVE:
        print("[Config] Remote/save mode: figure will be saved to all_filters_sample_N.png (no window)")
    print(f"[Config] Frame t = {FRAME_TO_SHOW}")

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
            continue
        wavL, wavR, x3, y = batch
        wavL = wavL.to(DEVICE).float()
        wavR = wavR.to(DEVICE).float()
        x3 = _sanitize_x3(x3.to(DEVICE))
        with torch.no_grad():
            YL, YR, QL, QR, XL, XR = model.bifb(wavL, wavR)
        QtL = QL[0].detach().cpu()
        QtR = QR[0].detach().cpu()
        frame_idx = min(max(0, FRAME_TO_SHOW), QtL.shape[0] - 1)
        current_sample = SAMPLE_IDX if SAMPLE_IDX is not None else n_plotted
        print(f"[Sample {current_sample}] Frame t={frame_idx}")

        fig = plot_three_bands_filters_lr_together(
            model=model, Q0=Q0, QtL=QtL, QtR=QtR, fcs=fcs, frame_idx=frame_idx,
            band_idxs=band_idxs, band_labels=band_labels, colors_dict=COLORS, sample_idx=current_sample,
        )
        if _SAVE_PLOT_TO_FILE or REMOTE_SAVE:
            out_dir = os.path.dirname(os.path.abspath(__file__))
            out_path = os.path.join(out_dir, f"all_filters_sample_{current_sample}.png")
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.01, transparent=True)
            print(f"[Saved] {out_path}")
            plt.close(fig)
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

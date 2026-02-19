"""
Plot subband energy spectrograms: Fixed Q0 (Initial) vs Adaptive Q(t).

Two subplots side-by-side (same size as Fig3_a):
  - Left:  log1p(Y) with fixed initial Q0 (no adaptive)
  - Right: log1p(Y) with adaptive Q(t)

Based on plot_Fig3_a.py (layout/size) and plot_q_filter_effect.py (spectrogram logic).
"""
#%%
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
if not os.environ.get("DISPLAY", "").strip():
    matplotlib.use("Agg")
else:
    for _backend in ("TkAgg", "Qt5Agg", "GTK4Agg", "WXAgg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False

from model_torch import build_model_active
from data import DeepEarH5Dataset_Active

# =========================
# USER CONFIG (align with plot_Fig3_a / plot_q_filter_effect)
# =========================
CKPT_PATH = "/media/mengh/SharedData/hanyu/DeepEar/runs_deepear_exps/active_ctrl-dual_fixedq-0_type-adaptiveQ_alpha0_cc-1_qctrlfrozen-0_bs64_lrfb0.0001_lrbe0.0001_wd1e-05_lossw0.20_0.40_0.40_run20260114-143653_dq1_lo0.3_hi5_rel_absolute-deltaq/checkpoints/best.pth"
TEST_H5 = "/media/mengh/SharedData/hanyu/DeepEar/anechoic_dataset/anechoic_test1_active_wav.h5"
# TEST_H5 = "/media/mengh/SharedData/hanyu/DeepEar/auditorium3_dataset/auditorium3_test1_active_wav.h5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CC = True
FB_ALPHA = 0
FIXED_FRONTEND_Q = False
DELTAQ_BASE = 1.0
DELTAQ_LOW_FACTOR = 0.3
DELTAQ_HIGH_FACTOR = 5.0
DELTAQ_MODE = "relative"

BATCH_SIZE = 1
SAMPLE_IDX = 12

# Plot style: same as Fig3_a (6, 2.5) for 1x2 subplots
FIG_SIZE = (6, 2.5)
DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 10

REMOTE_SAVE = True
OUT_NAME = "Q0_vs_adaptive_spectrogram.png"

EPS = 1e-12


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


def build_W_from_Q(model, Q_bn: torch.Tensor, device):
    f = model.bifb.f_fft.view(1, 1, -1).to(device)
    fc = model.bifb.fc.view(1, model.bifb.Nbands, 1).to(device)
    bw = (model.bifb.fc.unsqueeze(0).to(device) / (Q_bn + 1e-8)).unsqueeze(-1)
    W = torch.exp(-0.5 * ((f - fc) / (bw + 1e-12)) ** 2)
    W = torch.clamp(W, min=1e-12)
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-12)
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return W


@torch.no_grad()
def compute_Y_from_Xmag(model, Xmag_bf: torch.Tensor, Q_bn: torch.Tensor):
    device = Xmag_bf.device
    W = build_W_from_Q(model, Q_bn, device)
    Y = torch.einsum("bf,bnf->bn", Xmag_bf, W)
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return Y


@torch.no_grad()
def compute_Y_time_from_X(model, X_all_btf: torch.Tensor, Q_all_btn: torch.Tensor):
    B, T, _ = X_all_btf.shape
    Y_list = []
    for t in range(T):
        Xmag = X_all_btf[:, t, :].abs()
        Q_bn = Q_all_btn[:, t, :]
        Y_bn = compute_Y_from_Xmag(model, Xmag, Q_bn)
        Y_list.append(Y_bn)
    return torch.stack(Y_list, dim=1)


# =========================
# Main plot
# =========================
def main():
    assert os.path.exists(CKPT_PATH), f"CKPT not found: {CKPT_PATH}"
    assert os.path.exists(TEST_H5), f"Test H5 not found: {TEST_H5}"
    print(f"[Config] Device: {DEVICE}, Sample: {SAMPLE_IDX}")

    model = build_model_active(
        use_cc=USE_CC,
        fb_alpha=FB_ALPHA,
        fixed_frontend_q=bool(FIXED_FRONTEND_Q),
        deltaQ_base=DELTAQ_BASE,
        deltaQ_low_factor=DELTAQ_LOW_FACTOR,
        deltaQ_high_factor=DELTAQ_HIGH_FACTOR,
        deltaQ_mode=DELTAQ_MODE,
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = _load_state_dict_flexible(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] Loaded: {CKPT_PATH}")
    model.eval()

    ds = DeepEarH5Dataset_Active(TEST_H5)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Get the specified sample
    for batch_idx, batch in enumerate(loader):
        if batch_idx != SAMPLE_IDX:
            continue
        wavL, wavR, x3, y = batch
        wavL = wavL.to(DEVICE).float()
        wavR = wavR.to(DEVICE).float()
        x3 = _sanitize_x3(x3.to(DEVICE))
        break
    else:
        raise ValueError(f"Sample index {SAMPLE_IDX} not found (dataset has {len(ds)} samples)")

    # Run filterbank
    with torch.no_grad():
        YL_adapt, YR_adapt, QL, QR, XL, XR = model.bifb(wavL, wavR)

    Q0 = model.bifb.Q0.detach().cpu()
    T = YL_adapt.shape[1]
    Q0_all = Q0.view(1, 1, -1).repeat(1, T, 1).to(DEVICE)

    # Y with fixed Q0 vs adaptive Q(t)
    with torch.no_grad():
        YL_q0 = compute_Y_time_from_X(model, XL, Q0_all)[0].detach().cpu()
        YL_adapt_ = YL_adapt[0].detach().cpu()

    # Spectrogram-like: dB scale, 10*log10(1+Y), shape (T, Nbands)
    L0 = 10 * np.log10(1 + np.maximum(YL_q0.numpy(), 0))
    La = 10 * np.log10(1 + np.maximum(YL_adapt_.numpy(), 0))

    fcs = model.bifb.fc.detach().cpu().numpy()
    t_axis = np.arange(T)
    extent = [t_axis[0], t_axis[-1], fcs[0], fcs[-1]]

    # Shared color scale for both subplots
    vmin = min(L0.min(), La.min())
    vmax = max(L0.max(), La.max())

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': LABEL_SIZE - 1,
        'ytick.labelsize': LABEL_SIZE - 1,
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    })

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI, facecolor='white', sharey=True)

    cmap = "inferno"  # spectrogram: black->purple->red->yellow, good contrast
    im0 = axes[0].imshow(
        L0.T, origin="lower", aspect="auto", interpolation="nearest",
        extent=extent, vmin=vmin, vmax=vmax, cmap=cmap
    )
    axes[0].set_title("Passive (Left Ear)", fontweight='bold', pad=6)
    axes[0].set_xlabel("Frame t", fontweight='bold')
    axes[0].set_ylabel("Center frequency (Hz)", fontweight='bold')
    axes[0].set_xlim(t_axis[0], t_axis[-1])
    axes[0].set_ylim(fcs[0], fcs[-1])

    def fmt_freq(x, pos):
        if x >= 1000:
            return f"{int(x/1000)}k"
        return f"{int(x)}"

    for ax in axes:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_freq))

    im1 = axes[1].imshow(
        La.T, origin="lower", aspect="auto", interpolation="nearest",
        extent=extent, vmin=vmin, vmax=vmax, cmap=cmap
    )
    axes[1].set_title("Active (Left Ear)", fontweight='bold', pad=6)
    axes[1].set_xlabel("Frame t", fontweight='bold')
    axes[1].set_ylim(fcs[0], fcs[-1])

    plt.tight_layout(rect=[0.02, 0.03, 0.90, 0.96], pad=0.4, w_pad=0.8)
    cbar_ax = fig.add_axes([0.91, 0.08, 0.015, 0.84])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("dB", labelpad=2)

    if REMOTE_SAVE:
        out_dir = _SCRIPT_DIR
        out_path = os.path.join(out_dir, OUT_NAME)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03, transparent=True)
        print(f"[Saved] {out_path}")
    else:
        plt.show()
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()

# %%

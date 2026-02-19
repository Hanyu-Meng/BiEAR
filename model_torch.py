# model_torch.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Hyper-parameters (keep consistent with data.py)
# =========================
N_SECTORS = 8
N_DIST_CLASS = 5
DATA_DIM = 100
LATENT_DIM = 100

# =========================
# ERB utilities
# =========================
def erb_hz(f_hz: np.ndarray) -> np.ndarray:
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)

def erb_rate(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(4.37 * f_hz / 1000.0 + 1.0)

def inv_erb_rate(E: np.ndarray) -> np.ndarray:
    return (10 ** (E / 21.4) - 1.0) * 1000.0 / 4.37

def erb_spaced_fc_and_q(N=100, fmin=50.0, fmax=7200.0, erb_factor=1.019):
    Emin, Emax = erb_rate(fmin), erb_rate(fmax)
    E = np.linspace(Emin, Emax, N)
    fc = inv_erb_rate(E)                      # (N,)
    bw = erb_factor * erb_hz(fc)              # (N,)
    Q0 = fc / bw                              # (N,)
    return fc, Q0

def make_deltaQ_profile(
    fc_hz: torch.Tensor,
    deltaQ_base: float = 2.0,
    low_factor: float = 0.5,
    high_factor: float = 1.0
) -> torch.Tensor:
    fc_np = fc_hz.detach().cpu().numpy()
    E = erb_rate(fc_np)
    E = (E - E.min()) / (E.max() - E.min() + 1e-12)

    mult = low_factor + (high_factor - low_factor) * E
    mult = torch.tensor(mult, dtype=torch.float32, device=fc_hz.device)

    deltaQ_vec = deltaQ_base * mult
    deltaQ_vec = torch.clamp(deltaQ_vec, min=1e-3)
    return deltaQ_vec

# =========================
# AuralNet-style utilities
# =========================
def _build_sinusoidal_pos_encoding(T: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Standard Transformer sinusoidal positional encoding.
    Returns (T, d_model).
    """
    position = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)  # (T,1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                         (-math.log(10000.0) / max(d_model, 1)))
    pe = torch.zeros(T, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AuralNetGammatoneFB(nn.Module):
    """
    Fixed gammatone-like filterbank front-end used in AuralNet.

    This is an efficient ERB-based approximation implemented in the
    frequency domain (similar philosophy to the existing gammatone FB),
    with signal-processing parameters aligned to DeepEar active front-end
    (FramewiseAdaptiveGammatoneFB) by default:
      - fs = 16 kHz
      - n_bands = DATA_DIM (=100)
      - timesteps = 19 frames per 1 second
      - n_fft = 1024
      - fmax defaults to fs/2*0.9 (same as DeepEar code when fmax=None)

    Input:
        wav_1s: (B, Nsamp) waveform (~1 s)
    Output:
        Y: (B, T, Nbands) log-power-like energies per band and frame
    """
    def __init__(
        self,
        fs: int = 16000,
        n_bands: int = DATA_DIM,
        fmin: float = 50.0,
        fmax: float = None,
        timesteps: int = 19,
        hop_ratio: float = 1.0,
        n_fft: int = 1024,
    ):
        super().__init__()
        self.fs = fs
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.timesteps = int(timesteps)
        self.hop_ratio = float(hop_ratio)

        # Match DeepEar active framing: win = fs / timesteps (≈ 1s/19 ≈ 52.6ms), hop = win * hop_ratio
        if self.timesteps <= 0:
            raise ValueError(f"[AuralNetGammatoneFB] timesteps must be > 0, got {self.timesteps}")
        win = int(round(fs / self.timesteps))
        win = max(1, win)
        hop = int(round(win * self.hop_ratio))
        hop = max(1, hop)
        self.win = win
        self.hop = hop

        self.register_buffer("win_fn", torch.hann_window(self.win), persistent=False)

        Ffft = n_fft // 2 + 1
        f_fft = torch.linspace(0.0, fs / 2.0, Ffft)
        self.register_buffer("f_fft", f_fft)

        # Keep consistent with DeepEar: if fmax is None -> use fs/2*0.9
        if fmax is None:
            fmax = fs / 2.0 * 0.9

        fc_np, Q0_np = erb_spaced_fc_and_q(n_bands, fmin, fmax, erb_factor=1.019)
        self.register_buffer("fc", torch.tensor(fc_np, dtype=torch.float32))
        self.register_buffer("Q0", torch.tensor(Q0_np, dtype=torch.float32))

        self.Q_min, self.Q_max = 0.05, 30.0

    def _frame_1s(self, wav_1s: torch.Tensor) -> torch.Tensor:
        """
        Frame ~1 s waveform into overlapping windows, then make exactly `timesteps` frames
        (same behavior as FramewiseAdaptiveGammatoneFB).
        """
        B, Nsamp = wav_1s.shape
        target = self.fs

        if Nsamp < target:
            wav = F.pad(wav_1s, (0, target - Nsamp))
        else:
            wav = wav_1s[:, :target]

        if target < self.win:
            wav = F.pad(wav, (0, self.win - target))
            target = wav.shape[1]

        frames = wav.unfold(dimension=1, size=self.win, step=self.hop)  # (B, n_avail, win)
        n_avail = frames.shape[1]

        T = self.timesteps
        if n_avail >= T:
            frames = frames[:, :T, :]
        else:
            pad_T = T - n_avail
            frames = F.pad(frames, (0, 0, 0, pad_T))

        return frames  # (B,T,win)

    def forward(self, wav_1s: torch.Tensor) -> torch.Tensor:
        if wav_1s.dim() != 2:
            raise ValueError(f"[AuralNetGammatoneFB] Expected wav_1s (B,N), got {wav_1s.shape}")

        device = wav_1s.device
        frames = self._frame_1s(wav_1s)  # (B,T,win)
        B, T, _ = frames.shape

        win_fn = self.win_fn.to(device)
        f_fft = self.f_fft.to(device)
        fc = self.fc.to(device)
        Q0 = torch.clamp(self.Q0.to(device), self.Q_min, self.Q_max)  # (Nbands,)

        # Window and STFT
        frames_w = frames * win_fn  # (B,T,win)
        X = torch.fft.rfft(frames_w, n=self.n_fft)  # (B,T,Ffft) complex
        Xmag = X.abs()  # (B,T,Ffft)

        # Pre-compute fixed ERB-like filter responses W(f) for each band
        # Using Gaussian weighting in frequency domain as an approximation.
        bw = (fc / (Q0 + 1e-8)).unsqueeze(-1) + 1e-8  # (Nbands,1)
        f = f_fft.view(1, -1)  # (1,Ffft)
        fc_mat = fc.view(-1, 1)  # (Nbands,1)
        bw_mat = bw  # (Nbands,1)

        W = torch.exp(-0.5 * ((f - fc_mat) / bw_mat) ** 2)  # (Nbands,Ffft)
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
        W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)  # (Nbands,Ffft)

        # Apply filterbank over frequency axis
        # Xmag: (B,T,Ffft), W: (Nbands,Ffft) -> Y: (B,T,Nbands)
        Y = torch.einsum("btf,nf->btn", Xmag, W)
        Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        return Y  # (B,T,Nbands)

# =========================
# (A) Monaural adaptive FB (has controller)
# =========================
class FramewiseAdaptiveGammatoneFB(nn.Module):
    """
    Input:
        wav_1s: (B, Nsamp)  # 1s waveform
    Output:
        Y_all:  (B, T, Nbands)
        Q_all:  (B, T, Nbands)
        X_all:  (B, T, Ffft) complex
    """
    def __init__(
        self, fs, timesteps=19, n_fft=1024,
        Nbands=100, Q0=8.0,
        fmin=50.0, fmax=None,
        hop_ratio=1.0, alpha=0.2,
        deltaQ_base: float = 2.0,
        deltaQ_low_factor: float = 0.5,
        deltaQ_high_factor: float = 1.0,
        deltaQ_mode: str = "absolute"  # "absolute" or "relative"
    ):
        super().__init__()
        self.fs = fs
        self.timesteps = timesteps
        self.n_fft = n_fft
        self.Nbands = Nbands
        self.alpha = alpha

        win = int(round(fs / timesteps))
        hop = int(round(win * hop_ratio))
        hop = max(1, hop)
        self.win = win
        self.hop = hop

        self.register_buffer("win_fn", torch.hann_window(self.win), persistent=False)

        Ffft = n_fft // 2 + 1
        f_fft = torch.linspace(0, fs / 2, Ffft)
        self.register_buffer("f_fft", f_fft)

        if fmax is None:
            fmax = fs / 2 * 0.9

        fc_np, Q0_np = erb_spaced_fc_and_q(Nbands, fmin, fmax, erb_factor=1.019)
        self.register_buffer("fc", torch.tensor(fc_np, dtype=torch.float32))
        self.register_buffer("Q0", torch.tensor(Q0_np, dtype=torch.float32))

        deltaQ_vec = make_deltaQ_profile(
            fc_hz=self.fc,
            deltaQ_base=deltaQ_base,
            low_factor=deltaQ_low_factor,
            high_factor=deltaQ_high_factor
        )
        self.register_buffer("deltaQ_vec", deltaQ_vec)
        self.deltaQ_mode = deltaQ_mode.lower()  # "absolute" or "relative"

        # Controller
        # Original (deeper):
        self.q_rnn = nn.GRU(input_size=2 * Nbands, hidden_size=128, batch_first=True)
        self.q_out = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, Nbands),
        )
        # self.q_rnn = nn.GRU(input_size=2 * Nbands, hidden_size=64, batch_first=True)
        # self.q_out = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.LayerNorm(64),
        #     nn.SiLU(),
        #     nn.Linear(64, Nbands),
        # )

        # self.q_out = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.SiLU(),
        #     nn.Linear(128, Nbands),
        # )

        self.Q_min, self.Q_max = 0.05, 30.0
        self.freeze_Q = False  # if True -> Q always equals Q0

        nn.init.zeros_(self.q_out[-1].weight)
        nn.init.zeros_(self.q_out[-1].bias)

    def _frame_1s(self, wav_1s: torch.Tensor) -> torch.Tensor:
        B, Nsamp = wav_1s.shape
        target = self.fs

        if Nsamp < target:
            wav = F.pad(wav_1s, (0, target - Nsamp))
        else:
            wav = wav_1s[:, :target]

        if target < self.win:
            wav = F.pad(wav, (0, self.win - target))
            target = wav.shape[1]

        frames = wav.unfold(dimension=1, size=self.win, step=self.hop)  # (B, n_avail, win)

        n_avail = frames.shape[1]
        T = self.timesteps
        if n_avail >= T:
            frames = frames[:, :T, :]
        else:
            pad_T = T - n_avail
            frames = F.pad(frames, (0, 0, 0, pad_T))

        return frames  # (B,T,win)

    def forward(self, wav_1s: torch.Tensor):
        if wav_1s.dim() != 2:
            raise ValueError(f"Expected wav_1s (B,N), got {wav_1s.shape}")

        device = wav_1s.device
        frames = self._frame_1s(wav_1s)  # (B,T,win)
        B, T, _ = frames.shape

        Q_prev = self.Q0.unsqueeze(0).expand(B, -1)  # (B,N)
        h_q = None

        Y_all, Q_all, X_all = [], [], []

        f = self.f_fft.view(1, 1, -1).to(device)                 # (1,1,Ffft)
        fc = self.fc.view(1, self.Nbands, 1).to(device)          # (1,N,1)
        win_fn = self.win_fn.to(device)

        Y_prev_ctrl = torch.zeros(B, self.Nbands, device=device)

        for t in range(T):
            frame = frames[:, t, :] * win_fn
            X = torch.fft.rfft(frame, n=self.n_fft)              # (B,Ffft) complex
            X_all.append(X)

            Xmag = X.abs()

            bw = (self.fc.unsqueeze(0).to(device) / (Q_prev + 1e-8)).unsqueeze(-1) + 1e-8
            W = torch.exp(-0.5 * ((f - fc) / bw) ** 2)  # (B,N,Ffft)
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

            Y = torch.einsum("bf,bnf->bn", Xmag, W)
            Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

            Y_all.append(Y)
            Q_all.append(Q_prev)

            Y_ctrl = torch.log1p(torch.clamp(Y, min=0.0))   # (B,N)
            # q_feat = torch.cat([Y_ctrl, Y_prev_ctrl], dim=-1)  # (B,2N)
            # q_in = q_feat.unsqueeze(1)  # (B,1,2N)
            Y_prev_ctrl = Y_ctrl.detach()
            beta = 0.8  # 0~1，越大记忆越长
            Y_mem = torch.zeros(B, self.Nbands, device=device)
            Y_mem = beta * Y_mem + (1 - beta) * Y_ctrl.detach()
            q_feat = torch.cat([Y_ctrl, Y_mem], dim=-1)
            q_in = q_feat.unsqueeze(1) 

            if self.freeze_Q:
                Q_prev = self.Q0.unsqueeze(0).expand(B, -1)
                h_q = None
                Y_prev_ctrl = torch.zeros_like(Y_prev_ctrl)
            else:
                q_h, h_q = self.q_rnn(q_in, h_q)  # (B,1,128)
                delta = torch.tanh(self.q_out(q_h.squeeze(1)))    # (B,N) in [-1,1]
                
                if self.deltaQ_mode == "relative":
                    # Relative mode: Q = Q0 * (1 + deltaQ_vec * delta)
                    Q_prev = self.Q0.unsqueeze(0) * (1.0 + self.deltaQ_vec.unsqueeze(0) * delta)
                else:
                    # Absolute mode: Q = Q0 + deltaQ_vec * delta (default)
                    Q_prev = self.Q0.unsqueeze(0) + self.deltaQ_vec.unsqueeze(0) * delta
                
                Q_prev = torch.clamp(Q_prev, self.Q_min, self.Q_max)

                if not torch.isfinite(Q_prev).all():
                    Q_prev = self.Q0.unsqueeze(0).expand(B, -1)
                    h_q = None

        Y_all = torch.stack(Y_all, dim=1)
        Q_all = torch.stack(Q_all, dim=1)
        X_all = torch.stack(X_all, dim=1)  # complex

        return Y_all, Q_all, X_all

# =========================
# (A2) Monaural fixed FB (NO controller at all)
# =========================
class FramewiseFixedGammatoneFB(nn.Module):
    """
    Fixed front-end (no controller):
      - Q is always Q0 (ERB-based)
      - No q_rnn/q_out parameters exist
    """
    def __init__(self, fs, timesteps=19, n_fft=1024,
                 Nbands=100, fmin=50.0, fmax=None, hop_ratio=1.0):
        super().__init__()
        self.fs = fs
        self.timesteps = timesteps
        self.n_fft = n_fft
        self.Nbands = Nbands

        win = int(round(fs / timesteps))
        hop = int(round(win * hop_ratio))
        hop = max(1, hop)
        self.win = win
        self.hop = hop

        self.register_buffer("win_fn", torch.hann_window(self.win), persistent=False)

        Ffft = n_fft // 2 + 1
        f_fft = torch.linspace(0, fs / 2, Ffft)
        self.register_buffer("f_fft", f_fft)

        if fmax is None:
            fmax = fs / 2 * 0.9

        fc_np, Q0_np = erb_spaced_fc_and_q(Nbands, fmin, fmax, erb_factor=1.019)
        self.register_buffer("fc", torch.tensor(fc_np, dtype=torch.float32))
        self.register_buffer("Q0", torch.tensor(Q0_np, dtype=torch.float32))

        self.Q_min, self.Q_max = 0.05, 30.0

    def _frame_1s(self, wav_1s: torch.Tensor) -> torch.Tensor:
        B, Nsamp = wav_1s.shape
        target = self.fs

        if Nsamp < target:
            wav = F.pad(wav_1s, (0, target - Nsamp))
        else:
            wav = wav_1s[:, :target]

        if target < self.win:
            wav = F.pad(wav, (0, self.win - target))
            target = wav.shape[1]

        frames = wav.unfold(dimension=1, size=self.win, step=self.hop)  # (B, n_avail, win)

        n_avail = frames.shape[1]
        T = self.timesteps
        if n_avail >= T:
            frames = frames[:, :T, :]
        else:
            pad_T = T - n_avail
            frames = F.pad(frames, (0, 0, 0, pad_T))

        return frames  # (B,T,win)

    def forward(self, wav_1s: torch.Tensor):
        if wav_1s.dim() != 2:
            raise ValueError(f"Expected wav_1s (B,N), got {wav_1s.shape}")

        device = wav_1s.device
        frames = self._frame_1s(wav_1s)  # (B,T,win)
        B, T, _ = frames.shape

        Q0 = torch.clamp(self.Q0, self.Q_min, self.Q_max).to(device)  # (N,)
        Q_prev = Q0.unsqueeze(0).expand(B, -1)                         # (B,N)

        Y_all, Q_all, X_all = [], [], []

        f = self.f_fft.view(1, 1, -1).to(device)                 # (1,1,Ffft)
        fc = self.fc.view(1, self.Nbands, 1).to(device)          # (1,N,1)
        win_fn = self.win_fn.to(device)

        for t in range(T):
            frame = frames[:, t, :] * win_fn
            X = torch.fft.rfft(frame, n=self.n_fft)              # (B,Ffft) complex
            X_all.append(X)

            bw = (self.fc.unsqueeze(0).to(device) / (Q_prev + 1e-8)).unsqueeze(-1) + 1e-8
            W = torch.exp(-0.5 * ((f - fc) / bw) ** 2)           # (B,N,Ffft)
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

            Y = torch.einsum("bf,bnf->bn", X.abs(), W)
            Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

            Y_all.append(Y)
            Q_all.append(Q_prev)

        Y_all = torch.stack(Y_all, dim=1)  # (B,T,N)
        Q_all = torch.stack(Q_all, dim=1)  # (B,T,N)
        X_all = torch.stack(X_all, dim=1)  # (B,T,Ffft) complex
        return Y_all, Q_all, X_all

# =========================
# (B) Binaural FB: dual only
# =========================
class BinauralAdaptiveGammatoneFB(nn.Module):
    """
    Dual-only binaural FB (two independent monaural FBs).

    Two modes:
      - fixed_frontend_q=True : uses FramewiseFixedGammatoneFB (NO controller modules exist)
      - fixed_frontend_q=False: uses FramewiseAdaptiveGammatoneFB (controller exists, Q is adaptive)
    """
    def __init__(
        self,
        fs=16000,
        timesteps=19,
        n_fft=1024,
        Nbands=100,
        fmin=50.0,
        fmax=None,
        hop_ratio=1.0,
        alpha: float = 0.2,
        fixed_frontend_q: bool = False,
        deltaQ_base: float = 2.0,
        deltaQ_low_factor: float = 0.5,
        deltaQ_high_factor: float = 1.0,
        deltaQ_mode: str = "absolute"
    ):
        super().__init__()
        self.controller_mode = "dual"  # keep for compatibility with your training code
        self.fs = fs
        self.timesteps = timesteps
        self.n_fft = n_fft
        self.Nbands = Nbands
        self.alpha = float(alpha)

        # Provide buffers for downstream phase extraction (DeepEarActiveWaveform uses these)
        fc_np, Q0_np = erb_spaced_fc_and_q(
            Nbands, fmin,
            (fs / 2 * 0.9) if fmax is None else fmax,
            erb_factor=1.019
        )
        self.register_buffer("fc", torch.tensor(fc_np, dtype=torch.float32))
        self.register_buffer("Q0", torch.tensor(Q0_np, dtype=torch.float32))
        self.register_buffer("f_fft", torch.linspace(0, fs / 2, n_fft // 2 + 1))

        self.fixed_frontend_q = bool(fixed_frontend_q)

        if self.fixed_frontend_q:
            # IMPORTANT: no controller parameters exist in this branch
            self.fb_L = FramewiseFixedGammatoneFB(
                fs=fs, timesteps=timesteps, n_fft=n_fft,
                Nbands=Nbands, fmin=fmin, fmax=fmax, hop_ratio=hop_ratio
            )
            self.fb_R = FramewiseFixedGammatoneFB(
                fs=fs, timesteps=timesteps, n_fft=n_fft,
                Nbands=Nbands, fmin=fmin, fmax=fmax, hop_ratio=hop_ratio
            )
        else:
            self.fb_L = FramewiseAdaptiveGammatoneFB(
                fs=fs, timesteps=timesteps, n_fft=n_fft,
                Nbands=Nbands, Q0=8.0, fmin=fmin, fmax=fmax,
                hop_ratio=hop_ratio, alpha=alpha,
                deltaQ_base=deltaQ_base,
                deltaQ_low_factor=deltaQ_low_factor,
                deltaQ_high_factor=deltaQ_high_factor,
                deltaQ_mode=deltaQ_mode,
            )
            self.fb_R = FramewiseAdaptiveGammatoneFB(
                fs=fs, timesteps=timesteps, n_fft=n_fft,
                Nbands=Nbands, Q0=8.0, fmin=fmin, fmax=fmax,
                hop_ratio=hop_ratio, alpha=alpha,
                deltaQ_base=deltaQ_base,
                deltaQ_low_factor=deltaQ_low_factor,
                deltaQ_high_factor=deltaQ_high_factor,
                deltaQ_mode=deltaQ_mode,
            )

        # Keep a shared attribute for compatibility (some training code toggles model.bifb.freeze_Q)
        # For fixed_frontend_q=True, this does nothing (no controller anyway).
        self.freeze_Q = False

    def forward(self, wavL_1s: torch.Tensor, wavR_1s: torch.Tensor):
        YL, QL, XL = self.fb_L(wavL_1s)
        YR, QR, XR = self.fb_R(wavR_1s)
        return YL, YR, QL, QR, XL, XR


# =========================
# (B2) Binaural FB: single controller (one controller controls both ears)
# =========================
class BinauralAdaptiveGammatoneFB_SingleController(nn.Module):
    """
    Single-controller binaural FB: one shared controller drives Q for BOTH ears.
    - Controller input: concat([log1p(Y_L), Y_L_mem, log1p(Y_R), Y_R_mem]) -> (B, 4*Nbands)
    - Controller output: delta (B, Nbands) applied to both ears
    - Supports deltaQ_mode: "absolute" and "relative"
    """
    def __init__(
        self,
        fs=16000,
        timesteps=19,
        n_fft=1024,
        Nbands=100,
        fmin=50.0,
        fmax=None,
        hop_ratio=1.0,
        alpha: float = 0.2,
        fixed_frontend_q: bool = False,
        deltaQ_base: float = 2.0,
        deltaQ_low_factor: float = 0.5,
        deltaQ_high_factor: float = 1.0,
        deltaQ_mode: str = "absolute"
    ):
        super().__init__()
        self.controller_mode = "single"
        self.fs = fs
        self.timesteps = timesteps
        self.n_fft = n_fft
        self.Nbands = Nbands
        self.alpha = float(alpha)
        self.fixed_frontend_q = bool(fixed_frontend_q)
        self.deltaQ_mode = deltaQ_mode.lower()

        fc_np, Q0_np = erb_spaced_fc_and_q(
            Nbands, fmin,
            (fs / 2 * 0.9) if fmax is None else fmax,
            erb_factor=1.019
        )
        self.register_buffer("fc", torch.tensor(fc_np, dtype=torch.float32))
        self.register_buffer("Q0", torch.tensor(Q0_np, dtype=torch.float32))
        self.register_buffer("f_fft", torch.linspace(0, fs / 2, n_fft // 2 + 1))

        win = int(round(fs / timesteps))
        hop = max(1, int(round(win * hop_ratio)))
        self.win = win
        self.hop = hop
        self.register_buffer("win_fn", torch.hann_window(self.win), persistent=False)

        deltaQ_vec = make_deltaQ_profile(
            fc_hz=self.fc,
            deltaQ_base=deltaQ_base,
            low_factor=deltaQ_low_factor,
            high_factor=deltaQ_high_factor
        )
        self.register_buffer("deltaQ_vec", deltaQ_vec)
        self.Q_min, self.Q_max = 0.05, 30.0
        self.freeze_Q = False

        if not fixed_frontend_q:
            # Single controller: input = [Y_L_ctrl, Y_L_mem, Y_R_ctrl, Y_R_mem] -> 4*Nbands
            # Original (smaller):
            # self.q_rnn = nn.GRU(input_size=4 * Nbands, hidden_size=64, batch_first=True)
            # self.q_out = nn.Sequential(
            #     nn.Linear(64, 64),
            #     nn.LayerNorm(64),
            #     nn.SiLU(),
            #     nn.Linear(64, Nbands),
            # )
            self.q_rnn = nn.GRU(input_size=4 * Nbands, hidden_size=128, batch_first=True)
            self.q_out = nn.Sequential(
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(p=0.1),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(p=0.1),
                nn.Linear(128, Nbands),
            )
            nn.init.zeros_(self.q_out[-1].weight)
            nn.init.zeros_(self.q_out[-1].bias)
        else:
            self.q_rnn = None
            self.q_out = None

        if fixed_frontend_q:
            self.fb_L = FramewiseFixedGammatoneFB(
                fs=fs, timesteps=timesteps, n_fft=n_fft,
                Nbands=Nbands, fmin=fmin, fmax=fmax, hop_ratio=hop_ratio
            )
            self.fb_R = FramewiseFixedGammatoneFB(
                fs=fs, timesteps=timesteps, n_fft=n_fft,
                Nbands=Nbands, fmin=fmin, fmax=fmax, hop_ratio=hop_ratio
            )
        else:
            self.fb_L = None
            self.fb_R = None

    def _frame_1s(self, wav_1s: torch.Tensor) -> torch.Tensor:
        B, Nsamp = wav_1s.shape
        target = self.fs
        if Nsamp < target:
            wav = F.pad(wav_1s, (0, target - Nsamp))
        else:
            wav = wav_1s[:, :target]
        if target < self.win:
            wav = F.pad(wav, (0, self.win - target))
        frames = wav.unfold(1, self.win, self.hop)
        T = self.timesteps
        if frames.shape[1] >= T:
            frames = frames[:, :T, :]
        else:
            frames = F.pad(frames, (0, 0, 0, T - frames.shape[1]))
        return frames

    def forward(self, wavL_1s: torch.Tensor, wavR_1s: torch.Tensor):
        if self.fixed_frontend_q:
            YL, QL, XL = self.fb_L(wavL_1s)
            YR, QR, XR = self.fb_R(wavR_1s)
            return YL, YR, QL, QR, XL, XR

        device = wavL_1s.device
        frames_L = self._frame_1s(wavL_1s)
        frames_R = self._frame_1s(wavR_1s)
        B, T, _ = frames_L.shape

        Q_prev = self.Q0.unsqueeze(0).expand(B, -1)
        h_q = None
        Y_L_mem = torch.zeros(B, self.Nbands, device=device)
        Y_R_mem = torch.zeros(B, self.Nbands, device=device)

        f = self.f_fft.view(1, 1, -1).to(device)
        fc = self.fc.view(1, self.Nbands, 1).to(device)
        win_fn = self.win_fn.to(device)

        YL_all, YR_all, Q_all, XL_all, XR_all = [], [], [], [], []
        beta = 0.8

        for t in range(T):
            frame_L = frames_L[:, t, :] * win_fn
            frame_R = frames_R[:, t, :] * win_fn
            X_L = torch.fft.rfft(frame_L, n=self.n_fft)
            X_R = torch.fft.rfft(frame_R, n=self.n_fft)
            XL_all.append(X_L)
            XR_all.append(X_R)

            Xmag_L = X_L.abs()
            Xmag_R = X_R.abs()

            bw = (self.fc.unsqueeze(0).to(device) / (Q_prev + 1e-8)).unsqueeze(-1) + 1e-8
            W = torch.exp(-0.5 * ((f - fc) / bw) ** 2)
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

            Y_L = torch.einsum("bf,bnf->bn", Xmag_L, W)
            Y_R = torch.einsum("bf,bnf->bn", Xmag_R, W)
            Y_L = torch.nan_to_num(Y_L, nan=0.0, posinf=0.0, neginf=0.0)
            Y_R = torch.nan_to_num(Y_R, nan=0.0, posinf=0.0, neginf=0.0)

            YL_all.append(Y_L)
            YR_all.append(Y_R)
            Q_all.append(Q_prev)

            Y_L_ctrl = torch.log1p(torch.clamp(Y_L, min=0.0))
            Y_R_ctrl = torch.log1p(torch.clamp(Y_R, min=0.0))

            q_feat = torch.cat([Y_L_ctrl, Y_L_mem, Y_R_ctrl, Y_R_mem], dim=-1)
            q_in = q_feat.unsqueeze(1)

            if self.freeze_Q or self.q_rnn is None:
                Q_prev = self.Q0.unsqueeze(0).expand(B, -1)
                h_q = None
                Y_L_mem = torch.zeros_like(Y_L_mem)
                Y_R_mem = torch.zeros_like(Y_R_mem)
            else:
                q_h, h_q = self.q_rnn(q_in, h_q)
                delta = torch.tanh(self.q_out(q_h.squeeze(1)))

                if self.deltaQ_mode == "relative":
                    Q_prev = self.Q0.unsqueeze(0) * (1.0 + self.deltaQ_vec.unsqueeze(0) * delta)
                else:
                    Q_prev = self.Q0.unsqueeze(0) + self.deltaQ_vec.unsqueeze(0) * delta

                Q_prev = torch.clamp(Q_prev, self.Q_min, self.Q_max)
                if not torch.isfinite(Q_prev).all():
                    Q_prev = self.Q0.unsqueeze(0).expand(B, -1)
                    h_q = None

                Y_L_mem = beta * Y_L_mem + (1 - beta) * Y_L_ctrl.detach()
                Y_R_mem = beta * Y_R_mem + (1 - beta) * Y_R_ctrl.detach()

        YL = torch.stack(YL_all, dim=1)
        YR = torch.stack(YR_all, dim=1)
        Q = torch.stack(Q_all, dim=1)
        XL = torch.stack(XL_all, dim=1)
        XR = torch.stack(XR_all, dim=1)
        return YL, YR, Q, Q, XL, XR


class AuralNetAttentionBlock(nn.Module):
    """
    Lightweight self-attention encoder used for AuralNet-style
    feature aggregation.

    Input:  (B, T, d_in)
    Output: (B, T, d_model)
    """
    def __init__(
        self,
        d_in: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,d_in)
        """
        B, T, _ = x.shape
        device = x.device
        h = self.proj(x)  # (B,T,d_model)

        # Add sinusoidal positional encoding
        pe = _build_sinusoidal_pos_encoding(T, self.d_model, device)  # (T,d_model)
        h = h + pe.unsqueeze(0)  # (B,T,d_model)

        out = self.encoder(h)  # (B,T,d_model)
        return out

# =========================
# Encoders + Heads (unchanged)
# =========================
class ILDEncoder(nn.Module):
    def __init__(self, input_dim=DATA_DIM, hidden_dim=200, latent_dim=LATENT_DIM):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, latent_dim, batch_first=True)

    def forward(self, xL, xR):
        ild = xL - xR
        ild = torch.nan_to_num(ild, nan=0.0, posinf=0.0, neginf=0.0)
        ild = torch.clamp(ild, -10.0, 10.0)
        ild = self.in_norm(ild)

        h, _ = self.gru1(ild)
        h2, _ = self.gru2(h)

        z_ild = h2.mean(dim=1)
        z_ild = torch.nan_to_num(z_ild, nan=0.0, posinf=0.0, neginf=0.0)
        return z_ild

class IPDEncoder(nn.Module):
    def __init__(self, input_dim=DATA_DIM, hidden_dim=200, latent_dim=LATENT_DIM):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, latent_dim, batch_first=True)

    def forward(self, xL, xR):
        delta = xL - xR
        ipd = torch.atan2(torch.sin(delta), torch.cos(delta))
        ipd = torch.nan_to_num(ipd, nan=0.0, posinf=0.0, neginf=0.0)
        ipd = self.in_norm(ipd)

        h1, _ = self.gru1(ipd)
        h2, _ = self.gru2(h1)

        z_ipd = h2.mean(dim=1)
        z_ipd = torch.nan_to_num(z_ipd, nan=0.0, posinf=0.0, neginf=0.0)
        return z_ipd

class SubHead(nn.Module):
    def __init__(self, body_dim=200, n_dist_class=N_DIST_CLASS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(body_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.sound = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.aoa = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.dist = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, n_dist_class),
        )

    def forward(self, body_feat):
        h = self.shared(body_feat)
        sound_logit = self.sound(h)
        aoa_pred = torch.sigmoid(self.aoa(h))
        dist_logits = self.dist(h)
        return sound_logit, aoa_pred, dist_logits

class DeepEarTorchILD(nn.Module):
    def __init__(
        self,
        use_cc: bool = True,
        data_dim: int = DATA_DIM,
        latent_dim: int = LATENT_DIM,
        n_sectors: int = N_SECTORS,
        n_dist_class: int = N_DIST_CLASS
    ):
        super().__init__()
        self.use_cc = use_cc
        self.encoder_ild = ILDEncoder(input_dim=data_dim, hidden_dim=200, latent_dim=latent_dim)
        self.encoder_ipd = IPDEncoder(input_dim=data_dim, hidden_dim=200, latent_dim=latent_dim)

        if self.use_cc:
            self.cc_proj = nn.Linear(data_dim, latent_dim)

        feat_dim = 2 * latent_dim + (latent_dim if self.use_cc else 0)

        self.body = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.subheads = nn.ModuleList([SubHead(200, n_dist_class=n_dist_class) for _ in range(n_sectors)])

    def forward(self, x1, x2, x3, x4, x5):
        z_ild = self.encoder_ild(x1, x2)
        z_ipd = self.encoder_ipd(x4, x5)

        feats = [z_ild, z_ipd]
        if self.use_cc:
            feats.append(self.cc_proj(x3))

        feat_cat = torch.cat(feats, dim=-1)
        body = self.body(feat_cat)

        sound_list, aoa_list, dist_list = [], [], []
        for head in self.subheads:
            s_logit, a_pred, d_logit = head(body)
            sound_list.append(s_logit)
            aoa_list.append(a_pred)
            dist_list.append(d_logit)

        sound_logits = torch.cat(sound_list, dim=1)
        aoa_pred = torch.cat(aoa_list, dim=1)
        dist_logits = torch.stack(dist_list, dim=1)

        return sound_logits, aoa_pred, dist_logits

# =========================
# Active waveform model
# =========================
class DeepEarActiveWaveform(nn.Module):
    """
    wavL,wavR (B,N) -> binaural FB (dual only)
      -> log-energy + phase -> encoders -> heads
    """
    def __init__(
        self,
        fs=16000,
        timesteps=19,
        n_fft=1024,
        n_bands=DATA_DIM,
        use_cc=True,
        latent_dim=LATENT_DIM,
        n_sectors=N_SECTORS,
        n_dist_class=N_DIST_CLASS,
        fb_alpha: float = 0.2,
        fixed_frontend_q: bool = False,
        deltaQ_base: float = 2.0,
        deltaQ_low_factor: float = 0.5,
        deltaQ_high_factor: float = 1.0,
        deltaQ_mode: str = "absolute",
        bifb_class=None
    ):
        super().__init__()
        self.use_cc = use_cc
        self.fb_alpha = fb_alpha

        BifbCls = bifb_class if bifb_class is not None else BinauralAdaptiveGammatoneFB
        self.bifb = BifbCls(
            fs=fs,
            timesteps=timesteps,
            n_fft=n_fft,
            Nbands=n_bands,
            alpha=fb_alpha,
            fixed_frontend_q=bool(fixed_frontend_q),
            deltaQ_base=deltaQ_base,
            deltaQ_low_factor=deltaQ_low_factor,
            deltaQ_high_factor=deltaQ_high_factor,
            deltaQ_mode=deltaQ_mode,
        )

        self.encoder_ild = ILDEncoder(input_dim=n_bands, hidden_dim=200, latent_dim=latent_dim)
        self.encoder_ipd = IPDEncoder(input_dim=n_bands, hidden_dim=200, latent_dim=latent_dim)

        if self.use_cc:
            self.cc_proj = nn.Linear(n_bands, latent_dim)

        feat_dim = 2 * latent_dim + (latent_dim if self.use_cc else 0)

        self.body = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.subheads = nn.ModuleList([SubHead(200, n_dist_class=n_dist_class) for _ in range(n_sectors)])

        self.last_QL = None
        self.last_QR = None
        self.last_Q = None

    def _assert_finite(self, x, name):
        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            xmin = x.min().item() if x.numel() > 0 else float("nan")
            xmax = x.max().item() if x.numel() > 0 else float("nan")
            raise RuntimeError(f"[NaN/Inf] {name} has {bad} non-finite values. min={xmin} max={xmax}")

    def _subband_phase_from_X(self, X_all, Q_all, f_fft, fc, eps_mag=1e-3):
        B, T, Ffft = X_all.shape
        N = fc.numel()
        device = X_all.device

        f  = f_fft.view(1, 1, -1).to(device)
        fc = fc.view(1, N, 1).to(device)

        phases = []
        for t in range(T):
            Q = Q_all[:, t, :]  # (B,N)
            bw = (fc.to(device).reshape(1, -1) / (Q + 1e-8)).unsqueeze(-1) + 1e-8
            W = torch.exp(-0.5 * ((f - fc) / bw) ** 2)
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)

            X = X_all[:, t, :]
            Wc = torch.complex(W, torch.zeros_like(W))
            Z = torch.einsum("bnf,bf->bn", Wc, X)

            mag = torch.clamp(Z.abs(), min=eps_mag)
            Zn = Z / mag
            phase = torch.atan2(Zn.imag, Zn.real)
            phases.append(phase)

        return torch.stack(phases, dim=1)

    def forward(self, wavL_1s, wavR_1s, x3=None):
        wavL_1s = wavL_1s.float()
        wavR_1s = wavR_1s.float()

        YL, YR, QL, QR, XL, XR = self.bifb(wavL_1s, wavR_1s)

        self._assert_finite(YL, "YL")
        self._assert_finite(YR, "YR")
        self._assert_finite(QL, "QL")
        self._assert_finite(QR, "QR")

        self.last_QL = QL
        self.last_QR = QR
        self.last_Q = 0.5 * (QL + QR)

        x1 = torch.log(YL + 1e-8)
        x2 = torch.log(YR + 1e-8)
        x1 = torch.clamp(x1, -12.0, 12.0)
        x2 = torch.clamp(x2, -12.0, 12.0)

        phaseL = self._subband_phase_from_X(XL, QL, self.bifb.f_fft, self.bifb.fc)
        phaseR = self._subband_phase_from_X(XR, QR, self.bifb.f_fft, self.bifb.fc)

        z_ild = self.encoder_ild(x1, x2)
        z_ipd = self.encoder_ipd(phaseL, phaseR)

        feats = [z_ild, z_ipd]
        if self.use_cc:
            if x3 is None:
                x3 = torch.zeros(wavL_1s.size(0), DATA_DIM, device=wavL_1s.device)
            x3 = x3.float()
            feats.append(self.cc_proj(x3))

        feat_cat = torch.cat(feats, dim=-1)
        body = self.body(feat_cat)

        sound_list, aoa_list, dist_list = [], [], []
        for head in self.subheads:
            s_logit, a_pred, d_logit = head(body)
            sound_list.append(s_logit)
            aoa_list.append(a_pred)
            dist_list.append(d_logit)

        sound_logits = torch.cat(sound_list, dim=1)
        aoa_pred = torch.cat(aoa_list, dim=1)
        dist_logits = torch.stack(dist_list, dim=1)

        return sound_logits, aoa_pred, dist_logits


class AuralNetActiveWaveform(nn.Module):
    """
    AuralNet-style active waveform model.

    Important design choice for this project:
      - We keep the *training pipeline, losses, and output tasks identical*
        to the existing DeepEar active model:
            wavL, wavR, x3  ->  sound_logits, aoa_pred, dist_logits
      - Only the *front-end feature extraction* is replaced by an
        AuralNet-like pipeline:
            Fixed Gammatone-like filterbank (n_bands) + attention-based aggregation.
    """
    def __init__(
        self,
        fs: int = 16000,
        use_cc: bool = True,
        # Align AuralNet front-end signal-processing to DeepEar active defaults
        # (DATA_DIM=100, timesteps=19, n_fft=1024).
        n_bands: int = DATA_DIM,
        timesteps: int = 19,
        hop_ratio: float = 1.0,
        n_fft: int = 1024,
        d_model: int = 128,
        n_sectors: int = N_SECTORS,
        n_dist_class: int = N_DIST_CLASS,
    ):
        super().__init__()
        self.use_cc = use_cc

        # Fixed gammatone-style front-end for each ear (framing aligned to DeepEar)
        self.fb_L = AuralNetGammatoneFB(fs=fs, n_bands=n_bands, timesteps=timesteps, hop_ratio=hop_ratio, n_fft=n_fft)
        self.fb_R = AuralNetGammatoneFB(fs=fs, n_bands=n_bands, timesteps=timesteps, hop_ratio=hop_ratio, n_fft=n_fft)

        # Expose a "bifb" attribute for compatibility with existing helpers,
        # but we do not distinguish front/back-end parameters here.
        self.bifb = None

        # Attention blocks for left, right, and interaural (L-R) features
        self.attn_L = AuralNetAttentionBlock(d_in=n_bands, d_model=d_model)
        self.attn_R = AuralNetAttentionBlock(d_in=n_bands, d_model=d_model)
        self.attn_diff = AuralNetAttentionBlock(d_in=n_bands, d_model=d_model)

        # CC / additional features projection (uses existing DATA_DIM=100)
        self.cc_proj = nn.Linear(DATA_DIM, d_model) if self.use_cc else None

        # Aggregated feature dimension: [z_L, z_R, z_diff] + optional z_cc
        feat_dim = 3 * d_model + (d_model if self.use_cc else 0)

        # Reuse the same body + SubHead design as DeepEar
        self.body = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.subheads = nn.ModuleList([
            SubHead(200, n_dist_class=n_dist_class)
            for _ in range(n_sectors)
        ])

        # For compatibility with Q-regularization code in training loop
        self.last_Q = None

    def forward(self, wavL_1s: torch.Tensor, wavR_1s: torch.Tensor, x3: torch.Tensor = None):
        """
        Inputs:
            wavL_1s, wavR_1s: (B, N) waveform (approx. 1 s at 16 kHz)
            x3:                (B, DATA_DIM) auxiliary features (e.g., CC)
        Outputs:
            sound_logits: (B, N_SECTORS)
            aoa_pred:     (B, N_SECTORS)
            dist_logits:  (B, N_SECTORS, N_DIST_CLASS)
        """
        wavL_1s = wavL_1s.float()
        wavR_1s = wavR_1s.float()

        # Simple input normalization / clamping
        wavL_1s = torch.clamp(wavL_1s, -1.0, 1.0)
        wavR_1s = torch.clamp(wavR_1s, -1.0, 1.0)

        # 1) Gammatone-like filterbank front-end
        YL = self.fb_L(wavL_1s)  # (B,T,Nbands)
        YR = self.fb_R(wavR_1s)  # (B,T,Nbands)

        # Log compression and safety
        xL = torch.log(YL + 1e-8)
        xR = torch.log(YR + 1e-8)
        xL = torch.clamp(xL, -12.0, 12.0)
        xR = torch.clamp(xR, -12.0, 12.0)

        # Interaural difference feature
        xDiff = xL - xR

        # 2) Attention-based temporal aggregation
        hL = self.attn_L(xL)      # (B,T,d_model)
        hR = self.attn_R(xR)      # (B,T,d_model)
        hDiff = self.attn_diff(xDiff)  # (B,T,d_model)

        # Global average pooling over time dimension
        zL = hL.mean(dim=1)       # (B,d_model)
        zR = hR.mean(dim=1)       # (B,d_model)
        zDiff = hDiff.mean(dim=1) # (B,d_model)

        feats = [zL, zR, zDiff]

        if self.use_cc:
            if x3 is None:
                x3 = torch.zeros(wavL_1s.size(0), DATA_DIM, device=wavL_1s.device)
            x3 = x3.float()
            feats.append(self.cc_proj(x3))

        feat_cat = torch.cat(feats, dim=-1)  # (B, feat_dim)
        body = self.body(feat_cat)           # (B, 200)

        sound_list, aoa_list, dist_list = [], [], []
        for head in self.subheads:
            s_logit, a_pred, d_logit = head(body)
            sound_list.append(s_logit)
            aoa_list.append(a_pred)
            dist_list.append(d_logit)

        sound_logits = torch.cat(sound_list, dim=1)
        aoa_pred = torch.cat(aoa_list, dim=1)
        dist_logits = torch.stack(dist_list, dim=1)

        # last_Q left as None on purpose (no adaptive Q in this model)
        return sound_logits, aoa_pred, dist_logits

# =========================
# Builders
# =========================
def build_model(
    use_cc: bool = True,
    data_dim: int = DATA_DIM,
    latent_dim: int = LATENT_DIM,
    n_sectors: int = N_SECTORS,
    n_dist_class: int = N_DIST_CLASS
) -> nn.Module:
    return DeepEarTorchILD(
        use_cc=use_cc,
        data_dim=data_dim,
        latent_dim=latent_dim,
        n_sectors=n_sectors,
        n_dist_class=n_dist_class,
    )

def build_model_active_single_controller(
    use_cc=True,
    fs=16000,
    timesteps=19,
    n_fft=1024,
    data_dim=DATA_DIM,
    latent_dim=LATENT_DIM,
    n_sectors=N_SECTORS,
    n_dist_class=N_DIST_CLASS,
    fb_alpha: float = 0.2,
    fixed_frontend_q: bool = False,
    deltaQ_base: float = 2.0,
    deltaQ_low_factor: float = 0.5,
    deltaQ_high_factor: float = 1.0,
    deltaQ_mode: str = "absolute"
) -> nn.Module:
    """Active model with single shared controller (one controller for both ears)."""
    return DeepEarActiveWaveform(
        fs=fs,
        timesteps=timesteps,
        n_fft=n_fft,
        n_bands=data_dim,
        use_cc=use_cc,
        latent_dim=latent_dim,
        n_sectors=n_sectors,
        n_dist_class=n_dist_class,
        fb_alpha=fb_alpha,
        fixed_frontend_q=bool(fixed_frontend_q),
        deltaQ_base=deltaQ_base,
        deltaQ_low_factor=deltaQ_low_factor,
        deltaQ_high_factor=deltaQ_high_factor,
        deltaQ_mode=deltaQ_mode,
        bifb_class=BinauralAdaptiveGammatoneFB_SingleController,
    )


def build_model_active(
    use_cc=True,
    fs=16000,
    timesteps=19,
    n_fft=1024,
    data_dim=DATA_DIM,
    latent_dim=LATENT_DIM,
    n_sectors=N_SECTORS,
    n_dist_class=N_DIST_CLASS,
    fb_alpha: float = 0.2,
    fixed_frontend_q: bool = False,
    deltaQ_base: float = 2.0,
    deltaQ_low_factor: float = 0.5,
    deltaQ_high_factor: float = 1.0,
    deltaQ_mode: str = "absolute"
) -> nn.Module:
    return DeepEarActiveWaveform(
        fs=fs,
        timesteps=timesteps,
        n_fft=n_fft,
        n_bands=data_dim,
        use_cc=use_cc,
        latent_dim=latent_dim,
        n_sectors=n_sectors,
        n_dist_class=n_dist_class,
        fb_alpha=fb_alpha,
        fixed_frontend_q=bool(fixed_frontend_q),
        deltaQ_base=deltaQ_base,
        deltaQ_low_factor=deltaQ_low_factor,
        deltaQ_high_factor=deltaQ_high_factor,
        deltaQ_mode=deltaQ_mode,
    )


def build_model_auralnet_active(
    use_cc: bool = True,
    fs: int = 16000,
    n_bands: int = DATA_DIM,
    timesteps: int = 19,
    hop_ratio: float = 1.0,
    n_fft: int = 1024,
    d_model: int = 128,
    n_sectors: int = N_SECTORS,
    n_dist_class: int = N_DIST_CLASS,
) -> nn.Module:
    """
    Builder for the AuralNet-style active waveform model.

    This model is designed to be a drop-in replacement for the existing
    active model in your training script:
        - Same input interface:  wavL, wavR, x3
        - Same outputs:          sound_logits, aoa_pred, dist_logits
        - Same task definitions and loss functions.
    """
    return AuralNetActiveWaveform(
        fs=fs,
        use_cc=use_cc,
        n_bands=n_bands,
        timesteps=timesteps,
        hop_ratio=hop_ratio,
        n_fft=n_fft,
        d_model=d_model,
        n_sectors=n_sectors,
        n_dist_class=n_dist_class,
    )
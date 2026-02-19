import numpy as np
import librosa
import math
from gammatone.gtgram import gtgram
from scipy.signal import gammatone, lfilter, hilbert

DATA_DIM = 100
TIMESTEPS = 19

# ---------- 一些 ERB / gammatone 辅助函数 ----------

def hz2erb(f):
    """Hz -> ERB number (Glasberg & Moore)"""
    return 21.4 * np.log10(4.37e-3 * f + 1.0)

def erb2hz(e):
    """ERB number -> Hz"""
    return (10**(e / 21.4) - 1.0) / 4.37e-3

def erb_space(fmin, fmax, n_band):
    """在 ERB 轴上等间距取 n_band 个中心频率，然后映射回 Hz"""
    erb_min = hz2erb(fmin)
    erb_max = hz2erb(fmax)
    erb_points = np.linspace(erb_min, erb_max, n_band)
    return erb2hz(erb_points)   # (n_band,)

def build_gammatone_filterbank(fs, n_fft, n_band=DATA_DIM, fmin=50.0, fmax=None):
    """
    在 STFT 频率轴上构造一个 gammatone-like filterbank 矩阵：
    H: (n_band, n_freq) —— 行: 频带, 列: STFT 频点
    """
    if fmax is None:
        fmax = fs / 2.0

    # STFT 频率轴
    freqs = np.linspace(0, fs / 2.0, n_fft // 2 + 1)   # (n_freq,)

    # 100 个 gammatone 中心频率（ERB 等间距）
    centers = erb_space(fmin, fmax, n_band)           # (n_band,)

    H = np.zeros((n_band, freqs.size), dtype=np.float32)

    for i, fc in enumerate(centers):
        # ERB 带宽 (Hz)
        erb = 24.7 * (4.37e-3 * fc + 1.0)
        # 用一个高斯包络近似 gammatone 幅度响应（简单但好用）
        bw = 1.5 * erb   # 可以稍微宽一点
        H[i, :] = np.exp(-0.5 * ((freqs - fc) / bw)**2)

    # 每个滤波器归一化，使得在频率轴上权重和为 1
    H /= (H.sum(axis=1, keepdims=True) + 1e-8)
    return H   # (n_band, n_freq)

# ---------- 用 STFT + gammatone FB 得到 (TIMESTEPS, DATA_DIM) ----------
def compute_gammatone_feature_stft(wav_1d, fs,
                                   data_dim=DATA_DIM,
                                   timesteps=TIMESTEPS,
                                   fmin=50.0, fmax=None):
    """
    1) 对 wav_1d 做 STFT
    2) 用 gammatone-like filterbank 把 STFT 频谱整合为 data_dim 个频带
    3) 调整到 timesteps 帧, 返回 (timesteps, data_dim)
    """

    if fmax is None:
        fmax = fs / 2.0

    # ---------- 1. 设计 STFT 参数：大约 timesteps 帧 ----------
    N = len(wav_1d)
    # 不重叠，hop_length * timesteps ≈ N
    hop_length = max(1, N // timesteps)
    win_length = hop_length

    # n_fft 取 >= win_length 的最近 2 的幂
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    # ---------- 2. STFT ----------
    # 不居中，保证整段长度大致对应 timesteps 帧
    S = librosa.stft(
        wav_1d.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=False
    )  # (n_fft/2+1, T_frames)

    mag = np.abs(S)             # 幅度谱
    power = mag**2              # 功率谱 (n_freq, T_frames)

    # ---------- 3. 构造 gammatone-like 滤波器组 ----------
    H = build_gammatone_filterbank(fs, n_fft, n_band=data_dim,
                                   fmin=fmin, fmax=fmax)  # (data_dim, n_freq)

    # ---------- 4. 频带能量整合 ----------
    # band_energy: (data_dim, T_frames)
    band_energy = np.dot(H, power)

    # ---------- 5. 转成 dB，并对帧数做裁剪/补零 ----------
    # 防止 log(0)
    band_energy = np.maximum(band_energy, 1e-12)
    # 这里用 10*log10(功率)，数值大概在 [-40, 20] 这类
    gt_db = 10.0 * np.log10(band_energy)

    # 调整到固定 timesteps 帧
    T = gt_db.shape[1]
    if T < timesteps:
        pad = timesteps - T
        gt_db = np.pad(gt_db, ((0, 0), (0, pad)), mode="constant", constant_values=-80.0)
    elif T > timesteps:
        gt_db = gt_db[:, :timesteps]

    # 转成 (timesteps, data_dim)
    feat = gt_db.T.astype("float32")
    return feat

def compute_gammatone_feature(wav_1d, fs, data_dim=DATA_DIM, timesteps=TIMESTEPS):
    """
    对单通道波形做 gammatone 频谱：
      - 通道数 = data_dim
      - 时间帧数 ≈ timesteps（多了截断，少了补零）
    返回形状：(timesteps, data_dim)，可以直接喂给 GRU。
    """
    win_time = 1.0 / timesteps     # 粗糙地保证 1s 里大约 timesteps 帧
    hop_time = win_time            # 无重叠，这样 frames ≈ timesteps

    gt = gtgram(wav_1d, fs, win_time, hop_time,
                data_dim, 50.0)    # (channels, frames)
    gt = 20*np.log10(gt + 1e-8)
    # gt = gt - gt.max()
    # 调整到固定帧数 timesteps
    if gt.shape[1] < timesteps:
        pad = timesteps - gt.shape[1]
        gt = np.pad(gt, ((0, 0), (0, pad)), mode='constant')
    elif gt.shape[1] > timesteps:
        gt = gt[:, :timesteps]

    # 转成 (timesteps, data_dim)
    feat = gt.T.astype('float32')
    return feat

# 你之前的 ERB 工具可以复用
# hz2erb / erb2hz / erb_space 保持不变

def compute_gammatone_mag_phase_direct(
    wav_1d,
    fs,
    data_dim=DATA_DIM,
    timesteps=TIMESTEPS,
    fmin=50.0,
    fmax=None,
    order=4,
):
    """
    纯 gammatone 时域滤波 + Hilbert:
      - 不用 STFT
      - 直接得到每个 gammatone 频带的 mag(dB) + phase(rad)
    返回:
      feat_mag   : (timesteps, data_dim)
      feat_phase : (timesteps, data_dim)
    """

    if fmax is None:
        fmax = fs / 2.0

    wav_1d = np.asarray(wav_1d, dtype=np.float32)

    # ---------- 1) 设计 data_dim 个中心频率 ----------
    centers = erb_space(fmin, fmax, data_dim)
    centers = np.clip(centers, 1.0, fs/2 - 1.0)   # 关键修复：保证 < Nyquist

    N = len(wav_1d)

    # 按你 DeepEar 风格切帧：大约 timesteps 帧，不重叠
    frame_len = max(1, N // timesteps)
    hop = frame_len
    # 为简单起见，不额外对齐末尾，直接 pad 到整帧
    total_len = frame_len * timesteps
    if total_len > N:
        pad = total_len - N
        wav_pad = np.pad(wav_1d, (0, pad), mode="constant")
    else:
        wav_pad = wav_1d[:total_len]

    # 输出矩阵 (bands, frames)
    band_db    = np.zeros((data_dim, timesteps), dtype=np.float32)
    band_phase = np.zeros((data_dim, timesteps), dtype=np.float32)

    # ---------- 2) 对每个中心频率做一个 gammatone 滤波器 ----------
    for b, fc in enumerate(centers):
        # 2.1 设计 IIR gammatone 滤波器
        #  SciPy: b, a = gammatone(fc, 'iir', fs=fs, order=order)
        b_coefs, a_coefs = gammatone(fc, 'iir', fs=fs)

        # 2.2 过滤信号 → 子带信号
        subband = lfilter(b_coefs, a_coefs, wav_pad)   # (total_len,)

        # 2.3 analytic signal via Hilbert → mag & phase
        analytic = hilbert(subband)   # complex (total_len,)
        mag = np.abs(analytic)        # (total_len,)
        phase = np.angle(analytic)    # (total_len,)

        # 2.4 对每帧统计幅度 & 相位
        for t in range(timesteps):
            start = t * hop
            end   = start + frame_len
            frame_mag   = mag[start:end]
            frame_phase = phase[start:end]

            # 能量 / RMS → dB
            frame_mag = np.maximum(frame_mag, 1e-12)
            # 这里用 RMS 做一个 summary
            rms = np.sqrt(np.mean(frame_mag**2))
            db  = 20.0 * np.log10(rms + 1e-12)

            # 相位可以用“帧中心点”的相位，或者 circular mean
            center_idx = start + frame_len // 2
            if center_idx >= len(frame_phase):
                center_idx = len(frame_phase) - 1
            ph = frame_phase[center_idx]

            band_db[b, t]    = db
            band_phase[b, t] = ph

    # ---------- 3) 调整成 (timesteps, data_dim) ----------
    feat_mag   = band_db.T.astype(np.float32)    # (T, B)
    feat_phase = band_phase.T.astype(np.float32) # (T, B)

    return feat_mag, feat_phase

def stft_gammatone_mag_phase(
    wav_1d,
    fs,
    data_dim=DATA_DIM,
    timesteps=TIMESTEPS,
    fmin=50.0,
    fmax=None
):
    """
    返回:
      feat_mag   : (T, data_dim)  —— gammatone 频带能量 (dB)
      feat_phase : (T, data_dim)  —— gammatone 频带相位 (rad, -pi..pi)
    """

    if fmax is None:
        fmax = fs / 2.0

    # 1) 设计 STFT 参数
    N = len(wav_1d)
    hop_length = max(1, N // timesteps)
    win_length = hop_length

    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    # 2) 计算 STFT (复数谱) : (n_freq, T_frames)
    S = librosa.stft(
        wav_1d.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=False
    )

    n_freq, T_frames = S.shape

    # 3) 构造 gammatone FB
    H = build_gammatone_filterbank(fs, n_fft, n_band=data_dim,
                                   fmin=fmin, fmax=fmax)   # (data_dim, n_freq)

    # 4) 频带复数输出：band_complex (data_dim, T)
    #   每个 band: sum_f H(b,f) * S(f,t)
    band_complex = H @ S    # (B, T_frames)

    # 5) 从 band_complex 中取模长和相位
    band_mag = np.abs(band_complex)         # (B, T)
    band_phase = np.angle(band_complex)     # (B, T), rad

    # 6) 能量转 dB
    band_mag = np.maximum(band_mag, 1e-12)
    band_db = 20.0 * np.log10(band_mag)     # 你要振幅 dB 就用 20*log10

    # 7) 调整到固定帧数 timesteps
    T = band_db.shape[1]
    if T < timesteps:
        pad = timesteps - T
        band_db = np.pad(band_db, ((0, 0), (0, pad)),
                         mode="constant", constant_values=-80.0)
        band_phase = np.pad(band_phase, ((0, 0), (0, pad)),
                            mode="edge")   # phase 用边缘值填充
    elif T > timesteps:
        band_db = band_db[:, :timesteps]
        band_phase = band_phase[:, :timesteps]

    # 转成 (T, B)
    feat_mag   = band_db.T.astype("float32")
    feat_phase = band_phase.T.astype("float32")

    return feat_mag, feat_phase

def compute_gammatone_mag_phase(
    wav_1d,
    fs,
    data_dim=DATA_DIM,
    timesteps=TIMESTEPS,
    fmin=50.0,
    fmax=None
):
    """
    返回:
      feat_mag   : (timesteps, data_dim)  —— gammatone 幅度谱 (dB)
      feat_phase : (timesteps, data_dim)  —— gammatone 相位 (rad)
    """

    if fmax is None:
        fmax = fs / 2.0

    # =========================
    # 1) 设计 STFT 参数：保证 ≈ timesteps 帧
    # =========================
    N = len(wav_1d)
    hop_length = max(1, N // timesteps)
    win_length = hop_length

    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    # =========================
    # 2) STFT（复数）
    # =========================
    S = librosa.stft(
        wav_1d.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=False
    )   # (n_freq, T)

    # =========================
    # 3) 构造 gammatone filterbank
    # =========================
    H = build_gammatone_filterbank(
        fs, n_fft,
        n_band=data_dim,
        fmin=fmin,
        fmax=fmax
    )   # (data_dim, n_freq)

    # =========================
    # 4) 频带复数输出（最关键）
    # =========================
    band_complex = H @ S     # (data_dim, T)

    # =========================
    # 5) 幅度 & 相位
    # =========================
    band_mag   = np.abs(band_complex)          # amplitude
    band_phase = np.angle(band_complex)        # phase (-pi, pi)

    # 转 dB（注意：振幅用 20log10）
    band_mag = np.maximum(band_mag, 1e-10)
    band_db = 20.0 * np.log10(band_mag)

    # =========================
    # 6) 对齐到 timesteps
    # =========================
    T = band_db.shape[1]
    if T < timesteps:
        pad = timesteps - T
        band_db = np.pad(band_db, ((0, 0), (0, pad)),
                         mode="constant", constant_values=-80.0)
        band_phase = np.pad(band_phase, ((0, 0), (0, pad)),
                            mode="edge")
    elif T > timesteps:
        band_db = band_db[:, :timesteps]
        band_phase = band_phase[:, :timesteps]

    # 转成 (timesteps, data_dim)
    feat_mag   = band_db.T.astype("float32")
    feat_phase = band_phase.T.astype("float32")

    return feat_mag, feat_phase

def compute_cross_correlation_feature(left, right, fs, num_lags=DATA_DIM, max_lag_ms=3.0):
    """
    计算左右耳信号的互相关特征：
      - 物理范围：[-max_lag_ms, +max_lag_ms]
      - 输出维度：num_lags (例如 100)

    返回:
      cc_feat: (num_lags,)  float32
    """
    left = left.astype(np.float64)
    right = right.astype(np.float64)

    left -= left.mean()
    right -= right.mean()

    cc_full = np.correlate(left, right, mode="full")  # 长度 2N-1

    lags = np.arange(-len(left)+1, len(left)) / fs

    max_lag_sec = max_lag_ms * 1e-3
    mask = np.logical_and(lags >= -max_lag_sec, lags <= max_lag_sec)

    cc_crop = cc_full[mask]
    lags_crop = lags[mask]

    cc_crop /= (np.max(np.abs(cc_crop)) + 1e-8)

    target_lags = np.linspace(-max_lag_sec, max_lag_sec, num_lags)
    cc_resampled = np.interp(target_lags, lags_crop, cc_crop)

    return cc_resampled.astype(np.float32)   # (100,)
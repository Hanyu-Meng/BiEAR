import os
import glob
import numpy as np
import soundfile as sf
import pysofaconventions as sofa
from scipy.signal import fftconvolve, resample_poly
from tqdm import tqdm

# =========================
# 0. 配置参数（DeepEar 式 anechoic 数据）
# =========================

SOFA_FILE = "/media/mengh/SharedData/hanyu/DeepEar/TU_Berlin/QU_KEMAR_anechoic.sofa"
TIMIT_ROOT = "/media/mengh/SharedData/hanyu/AuralNet/TIMIT"  # 包含 TRAIN / TEST
OUT_ROOT = "/media/mengh/SharedData/hanyu/DeepEar/anechoic_dataset"

FS_TARGET = 16000
SEGMENT_SEC = 1.0
MAX_SOURCES = 3
N_SECTORS = 8
DIST_CLASSES = np.array([0.5, 1.0, 2.0, 3.0])

# DeepEar Table 2 样本数（可以先改成小一点测试）
DATASET_SPECS = {
    "anechoic_train":   {"num": 72000, "split": "TRAIN"},
    "anechoic_val":     {"num": 9000,  "split": "TRAIN"},
    "anechoic_test1":   {"num": 9000,  "split": "TRAIN"},
    "anechoic_test2":   {"num": 9000,  "split": "TEST"},
}

os.makedirs(OUT_ROOT, exist_ok=True)

# =========================
# 1. 读取 SOFA：HRIR + 方向 + 距离
# =========================

hrtf = sofa.SOFAFile(SOFA_FILE, 'r')
IR = hrtf.getDataIR()                       # (M, 2, N)
fs_hrir = float(hrtf.getSamplingRate())     # 44100
source_pos = hrtf.getVariableValue("SourcePosition")  # (M, 3): [az, el, dist]

M, R, N = IR.shape
print("========== SOFA INFO ==========")
print(f"Measurements: {M}")
print(f"Receivers: {R} (2 = 左/右耳)")
print(f"Samples per IR: {N}")
print(f"HRIR fs: {fs_hrir} Hz")
print("================================")

def resample_hrir_to_target(IR, fs_hrir, fs_target):
    """把 HRIR 从 44.1 kHz 重采样到 fs_target（16k）。"""
    if fs_hrir == fs_target:
        return IR
    M, R, N = IR.shape
    gcd = np.gcd(int(fs_hrir), int(fs_target))
    up = int(fs_target // gcd)
    down = int(fs_hrir // gcd)
    IR_rs = []
    for m in range(M):
        chs = []
        for r in range(R):
            ir_rs = resample_poly(IR[m, r, :], up, down)
            chs.append(ir_rs)
        IR_rs.append(np.stack(chs, axis=0))
    return np.stack(IR_rs, axis=0)


# 预先重采样到 16k
IR_16k = resample_hrir_to_target(IR, fs_hrir, FS_TARGET)
fs_hrir_rs = FS_TARGET
print(f"Resampled HRIR to {fs_hrir_rs} Hz, shape = {IR_16k.shape}")

# =========================
# 2. 预先按 sector 把 BRIR 索引分组
#    保证：1 个 sector 里最多 1 个 source，AoA 在该 sector 内随机
# =========================

az_all = source_pos[:, 0]  # degrees
sector_width = 360.0 / N_SECTORS
sector_indices = []
for sid in range(N_SECTORS):
    start = sid * sector_width
    end = start + sector_width
    idx = np.where((az_all >= start) & (az_all < end))[0]
    if len(idx) == 0:
        raise RuntimeError(f"No BRIRs found for sector {sid} [{start},{end})")
    sector_indices.append(idx)

# =========================
# 3. 工具函数
# =========================

def load_random_clean_segment(wav_paths, length_sec, fs_target):
    """随机选一条干净语音，抽一段 >= length_sec 的片段，重采样成 fs_target。"""
    while True:
        wav_path = np.random.choice(wav_paths)
        x, fs = sf.read(wav_path)
        if x.ndim > 1:
            x = x[:, 0]
        if fs != fs_target:
            gcd = np.gcd(fs, fs_target)
            up = fs_target // gcd
            down = fs // gcd
            x = resample_poly(x, up, down)
            fs = fs_target
        if len(x) >= int((length_sec + 0.2) * fs):
            max_start = len(x) - int(length_sec * fs)
            start = np.random.randint(0, max_start + 1)
            seg = x[start:start + int(length_sec * fs)]
            return seg, fs


def aoa_to_sector_label(az_deg, n_sectors=N_SECTORS):
    """
    AoA (0-360) -> sector_id + sector 内归一化角度 (0,1]。
    """
    az = az_deg % 360.0
    sector_width = 360.0 / n_sectors
    sector_id = int(np.floor(az / sector_width))
    in_sector_angle = az - sector_id * sector_width
    eps = 1e-3
    norm_angle = (in_sector_angle + eps) / sector_width
    norm_angle = min(norm_angle, 1.0)
    return sector_id, norm_angle


def distance_to_class_index(dist):
    """把实际距离映射到最近的 {0.5,1,2,3} 类别下标。"""
    idx = int(np.argmin(np.abs(DIST_CLASSES - dist)))
    return idx


# =========================
# 4. 收集 TIMIT 语音
# =========================

def collect_timit_wavs(split):
    """
    split: "TRAIN" 或 "TEST"
    """
    base = os.path.join(TIMIT_ROOT, split)
    wavs = sorted(
        glob.glob(os.path.join(base, "**", "*.WAV"), recursive=True)
        + glob.glob(os.path.join(base, "**", "*.wav"), recursive=True)
    )
    print(f"[{split}] found {len(wavs)} wavs.")
    if len(wavs) == 0:
        raise RuntimeError(f"No wavs found in {base}")
    return wavs


# =========================
# 5. 合成单个样本（给定某个 wav 列表）
# =========================

def synthesize_one_sample(clean_wavs, n_sources=None):
    """
    生成一条 1 秒双耳样本：
    - 如果 n_sources is None: 源个数 ~ Uniform{1,2,3}（训练/验证）
    - 如果 n_sources 给定：固定为 1 或 2 或 3（测试集可控）
    - 每个源占据不同 sector（最多 1 个源 / sector）
    """
    if n_sources is None:
        n_sources = np.random.choice([1, 2, 3])
    else:
        assert n_sources in [1, 2, 3]

    T = int(SEGMENT_SEC * FS_TARGET)

    y_left = np.zeros(T)
    y_right = np.zeros(T)

    labels = {
        "num_sources": n_sources,
        "aoa_deg": [],
        "distance_m": [],
        "sector_id": [],
        "sector_angle_norm": [],
        "distance_class": [],
    }

    # 为本样本选 n_sources 个不重复的 sector
    chosen_sectors = np.random.choice(np.arange(N_SECTORS),
                                      size=n_sources,
                                      replace=False)

    for sid in chosen_sectors:
        # 在该 sector 内随机选一个 BRIR 索引
        candidate_indices = sector_indices[sid]
        idx = np.random.choice(candidate_indices)
        irL = IR_16k[idx, 0, :]
        irR = IR_16k[idx, 1, :]
        az, el, dist = source_pos[idx]

        # 取一段干净语音
        x_seg, fs = load_random_clean_segment(clean_wavs, SEGMENT_SEC, FS_TARGET)
        assert fs == FS_TARGET

        # 卷积
        yL_full = fftconvolve(x_seg, irL)
        yR_full = fftconvolve(x_seg, irR)

        # 截取 1 秒
        if len(yL_full) <= T:
            yL = np.zeros(T)
            yR = np.zeros(T)
            yL[:len(yL_full)] = yL_full
            yR[:len(yR_full)] = yR_full
        else:
            max_start = len(yL_full) - T
            start = np.random.randint(0, max_start + 1)
            yL = yL_full[start:start + T]
            yR = yR_full[start:start + T]

        # 叠加
        y_left += yL
        y_right += yR

        # 标签
        sector_id, norm_ang = aoa_to_sector_label(az)
        dist_cls = distance_to_class_index(dist)

        labels["aoa_deg"].append(float(az))
        labels["distance_m"].append(float(dist))
        labels["sector_id"].append(int(sector_id))
        labels["sector_angle_norm"].append(float(norm_ang))
        labels["distance_class"].append(int(dist_cls))

    # 归一化到 [-1,1]
    max_abs = max(np.max(np.abs(y_left)), np.max(np.abs(y_right)), 1e-8)
    y_left /= max_abs
    y_right /= max_abs

    y = np.stack([y_left, y_right], axis=0)  # (2,T)
    return y, labels


# =========================
# 6. 生成四个 anechoic 数据集
# =========================

def generate_dataset(ds_name, num_samples, split):
    """
    ds_name: "anechoic_train" / "anechoic_val" / "anechoic_test1" / "anechoic_test2"
    split: "TRAIN" or "TEST" (TIMIT 部分)
    """
    print(f"\n=== Generating {ds_name} ({num_samples} samples, TIMIT {split}) ===")
    clean_wavs = collect_timit_wavs(split)

    out_dir = os.path.join(OUT_ROOT, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    # 对 test 数据集：前 1/3 → 1 源，中间 1/3 → 2 源，最后 1/3 → 3 源
    is_test = ds_name.startswith("anechoic_test")
    third = num_samples // 3

    for i in tqdm(range(num_samples), desc=ds_name):
        if is_test:
            if i < third:
                n_src = 1
            elif i < 2 * third:
                n_src = 2
            else:
                n_src = 3
        else:
            n_src = None  # train/val 还是 Uniform{1,2,3}

        y, labels = synthesize_one_sample(clean_wavs, n_sources=n_src)

        wav_path = os.path.join(out_dir, f"{ds_name}_{i:06d}.wav")
        sf.write(wav_path, y.T, FS_TARGET)

        npz_path = os.path.join(out_dir, f"{ds_name}_{i:06d}.npz")
        np.savez(
            npz_path,
            audio_path=wav_path,
            num_sources=labels["num_sources"],
            aoa_deg=np.array(labels["aoa_deg"], dtype=np.float32),
            distance_m=np.array(labels["distance_m"], dtype=np.float32),
            sector_id=np.array(labels["sector_id"], dtype=np.int64),
            sector_angle_norm=np.array(labels["sector_angle_norm"], dtype=np.float32),
            distance_class=np.array(labels["distance_class"], dtype=np.int64),
        )

    print(f"Finished {ds_name}, saved to {out_dir}")


def main():
    for ds_name, spec in DATASET_SPECS.items():
        generate_dataset(ds_name, spec["num"], spec["split"])

    print("\n✅ All anechoic datasets generated under:", OUT_ROOT)


if __name__ == "__main__":
    main()
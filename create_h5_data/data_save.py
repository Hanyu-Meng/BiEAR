# =========================
# NEW: 从 .npz + .wav 构造 gammatone 特征 & 标签
# =========================
from gammatone.gtgram import gtgram  # pip install gammatone
import os, glob
from tqdm import tqdm
import numpy as np
import soundfile as sf
import torch
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from torch.utils.data import Dataset
from utils_save import *
DATA_DIM = 100   # 对应原来的 data_dim
TIMESTEPS = 19   # 对应原来的 timesteps
N_SECTORS = 8    # 和你前面一致
N_DIST_CLASSES = 5  # 4 个距离 + 1 个 no-source
# ===== 按原来 DeepEar 的切法把 y 拆成 24 个输出 =====
def _worker_build_sample(npz_path):
    """
    单个 worker 进程执行的函数：
    给一个 npz -> 返回 (x1, x2, x3, y)
    """
    return build_sample_from_npz(npz_path)

def split_y_matrix(y_mat):
    """
    y_mat: (N,56)
    返回 24 个 y*_train / y*_val / y*_test
    每 7 维 = [sound, angle, dist(5)]
    """
    y1  = y_mat[:, 0]      # sound1
    y2  = y_mat[:, 1]      # angle1
    y3  = y_mat[:, 2:7]    # dist1 (5) --> whether we can only use 4 categories??

    y4  = y_mat[:, 7]
    y5  = y_mat[:, 8]
    y6  = y_mat[:, 9:14]

    y7  = y_mat[:, 14]
    y8  = y_mat[:, 15]
    y9  = y_mat[:, 16:21]

    y10 = y_mat[:, 21]
    y11 = y_mat[:, 22]
    y12 = y_mat[:, 23:28]

    y13 = y_mat[:, 28]
    y14 = y_mat[:, 29]
    y15 = y_mat[:, 30:35]

    y16 = y_mat[:, 35]
    y17 = y_mat[:, 36]
    y18 = y_mat[:, 37:42]

    y19 = y_mat[:, 42]
    y20 = y_mat[:, 43]
    y21 = y_mat[:, 44:49]

    y22 = y_mat[:, 49]
    y23 = y_mat[:, 50]
    y24 = y_mat[:, 51:56]

    return [y1, y2, y3,
            y4, y5, y6,
            y7, y8, y9,
            y10, y11, y12,
            y13, y14, y15,
            y16, y17, y18,
            y19, y20, y21,
            y22, y23, y24]


def build_label_from_npz_dict(d):
    """
    把 .npz 里的标签字段:
      - num_sources
      - aoa_deg
      - distance_m
      - sector_id
      - sector_angle_norm
      - distance_class (0..3, 对应 4 个距离)
    映射成长度 56 的向量 y：
      8 个 sector，每个 sector:
        [sound_presence, angle_norm, 5-d one-hot 距离]
      5-d 里 index 0 表示 no-source，其余 1..4 对应四个距离类。
    """
    num_sources = int(d["num_sources"])
    sector_ids = np.array(d["sector_id"], dtype=np.int64)
    angle_norms = np.array(d["sector_angle_norm"], dtype=np.float32)
    dist_classes = np.array(d["distance_class"], dtype=np.int64)  # 0..3

    sound = np.zeros(N_SECTORS, dtype=np.float32)
    angle = np.zeros(N_SECTORS, dtype=np.float32)
    dist_onehot = np.zeros((N_SECTORS, N_DIST_CLASSES), dtype=np.float32)

    # 先全部设为 no-source
    dist_onehot[:, 0] = 1.0

    for k in range(num_sources):
        sid = int(sector_ids[k])
        if sid < 0 or sid >= N_SECTORS:
            continue
        sound[sid] = 1.0
        angle[sid] = float(angle_norms[k])
        # 改写 one-hot：index 0 留给 no-source，1..4 对应四个距离
        dcls = int(dist_classes[k]) + 1  # 1..4
        dist_onehot[sid, :] = 0.0
        dist_onehot[sid, dcls] = 1.0

    # 拼成长度 56 的向量：按 sector1..8 依次 [sound, angle, dist_onehot(5)]
    y_vec = []
    for sid in range(N_SECTORS):
        y_vec.append(sound[sid])
        y_vec.append(angle[sid])
        y_vec.extend(dist_onehot[sid].tolist())
    y_vec = np.array(y_vec, dtype=np.float32)  # (56,)
    return y_vec


def build_sample_from_npz(npz_path):
    """
    给一个 .npz 文件：
      - 读标签
      - 读对应的 .wav
      - 做左右耳 gammatone 特征
      - 生成 x1, x2, x3, y
    x1, x2 形状：(TIMESTEPS, DATA_DIM)
    x3 是 100 维向量（我们现在没用 CC 特征，就填 0）
    """
    d = np.load(npz_path, allow_pickle=True)

    wav_path = str(d["audio_path"])
    audio, fs = sf.read(wav_path)   # (T, 2)
    if audio.ndim == 1:
        # 理论上应该是双通道，这里防一下万一
        left = audio
        right = audio
    else:
        left = audio[:, 0]
        right = audio[:, 1]

    # feat_L = compute_gammatone_feature(left, fs)   # (19,100)
    # feat_R = compute_gammatone_feature(right, fs)  # (19,100)
    # feat_L = compute_gammatone_feature_stft(left, fs)   # (19,100)
    # feat_R = compute_gammatone_feature_stft(right, fs)  # (19,100)
    # feat_L_mag, feat_L_phase = compute_gammatone_mag_phase_direct(left, fs)
    # feat_R_mag, feat_R_phase = compute_gammatone_mag_phase_direct(right, fs)

    x1 = left
    x2 = right
    # ✅ Cross-correlation (核心修复点)
    x3 = compute_cross_correlation_feature(
        left, right,
        fs=fs,
        num_lags=DATA_DIM,
        max_lag_ms=3.0
    )   # (100,)
    # x4 = feat_L_phase
    # x5 = feat_R_phase
    y_vec = build_label_from_npz_dict(d)
    # return x1, x2, x3, x4, x5, y_vec
    return x1, x2, x3, y_vec


# def load_dataset_from_dir(dataset_dir, max_samples=None):
#     """
#     扫描某个数据集目录（如 anechoic_train）,
#     读取所有 .npz 文件并构造：
#       x1: (N, TIMESTEPS, DATA_DIM)
#       x2: (N, TIMESTEPS, DATA_DIM)
#       x3: (N, DATA_DIM)
#       y:  (N, 56)
#     """
#     npz_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
#     if max_samples is not None:
#         npz_files = npz_files[:max_samples]
#     print(f"[load_dataset_from_dir] {dataset_dir}, found {len(npz_files)} npz files")

#     x1_list, x2_list, x3_list, y_list = [], [], [], []
#     for f in tqdm(npz_files, desc=f"Loading {os.path.basename(dataset_dir)}"):
#         featL, featR, cc_feat, y_vec = build_sample_from_npz(f)
#         x1_list.append(featL)
#         x2_list.append(featR)
#         x3_list.append(cc_feat)
#         y_list.append(y_vec)

#     x1 = np.stack(x1_list, axis=0)   # (N,19,100)
#     x2 = np.stack(x2_list, axis=0)
#     x3 = np.stack(x3_list, axis=0)   # (N,100)
#     y = np.stack(y_list, axis=0)     # (N,56)
#     return x1, x2, x3, y

def load_dataset_from_dir(dataset_dir, max_samples=None, num_workers=None):
    """
    ✅ 多进程加速版：
    扫描某个数据集目录（如 anechoic_train）,
    并行读取所有 .npz 并构造：
      x1: (N, TIMESTEPS, DATA_DIM)
      x2: (N, TIMESTEPS, DATA_DIM)
      x3: (N, DATA_DIM)
      y:  (N, 56)
    """

    npz_files = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))
    if max_samples is not None:
        npz_files = npz_files[:max_samples]

    print(f"[load_dataset_from_dir | multiprocess] {dataset_dir}, found {len(npz_files)} npz files")

    # ✅ 默认用 CPU 核心数 - 1
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Using {num_workers} worker processes")

    x1_list, x2_list, x3_list, y_list = [], [], [], []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_build_sample, f) for f in npz_files]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Processing {os.path.basename(dataset_dir)}"):
            featL, featR, cc_feat, y_vec = future.result()
            x1_list.append(featL)
            x2_list.append(featR)
            x3_list.append(cc_feat)
            y_list.append(y_vec)

    x1 = np.stack(x1_list, axis=0)   # (N,19,100)
    x2 = np.stack(x2_list, axis=0)
    x3 = np.stack(x3_list, axis=0)   # (N,100)
    y  = np.stack(y_list, axis=0)    # (N,56)

    return x1, x2, x3, y


class DeepEarH5Dataset(Dataset):
    """
    直接从 .h5 里按 index 读取:
      - x1: (TIMESTEPS, DATA_DIM)
      - x2: (TIMESTEPS, DATA_DIM)
      - x3: (DATA_DIM,)
      - y:  (56,)

    preload:
      - False: 按需从 h5 读（节省内存）
      - True:  一次性读进内存 (numpy 数组)，训练 I/O 速度最快
    """
    def __init__(self, h5_path, preload=False):
        super().__init__()
        self.h5_path = h5_path
        self.preload = preload
        self._h5 = None  # 延迟打开, 兼容多进程 DataLoader

        with h5py.File(h5_path, "r") as f:
            self.length = f["x1"].shape[0]
            assert f["x2"].shape[0] == self.length
            assert f["x3"].shape[0] == self.length
            assert f["x4"].shape[0] == self.length
            assert f["x5"].shape[0] == self.length
            assert f["y"].shape[0]  == self.length

            if preload:
                print(f"[DeepEarH5Dataset] Preloading {h5_path} into RAM ...")
                self.x1 = f["x1"][:].astype("float32")
                self.x2 = f["x2"][:].astype("float32")
                self.x3 = f["x3"][:].astype("float32")
                self.x4 = f["x4"][:].astype("float32")
                self.x5 = f["x5"][:].astype("float32")
                self.y  = f["y"][:].astype("float32")
                print(f"[DeepEarH5Dataset] Preload done. N = {self.length}")
            else:
                self.x1 = None
                self.x2 = None
                self.x3 = None
                self.x4 = None
                self.x5 = None
                self.y  = None

    def _get_h5(self):
        if self._h5 is None:
            # swmr=True 只读多进程更安全
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preload:
            x1 = self.x1[idx]
            x2 = self.x2[idx]
            x3 = self.x3[idx]
            x4 = self.x4[idx]
            x5 = self.x5[idx]
            y  = self.y[idx]
        else:
            f = self._get_h5()
            x1 = f["x1"][idx]   # (TIMESTEPS, DATA_DIM)
            x2 = f["x2"][idx]
            x3 = f["x3"][idx]   # (DATA_DIM,)
            x4 = f["x4"][idx]
            x5 = f["x5"][idx]
            y  = f["y"][idx]    # (56,)

        # 转成 torch.Tensor
        x1 = torch.from_numpy(x1)      # float32
        x2 = torch.from_numpy(x2)
        x3 = torch.from_numpy(x3)
        x4 = torch.from_numpy(x4)
        x5 = torch.from_numpy(x5)
        y  = torch.from_numpy(y)

        return x1, x2, x3, x4, x5, y

def load_arrays_from_h5(h5_path):
    """
    如果你想一次性把 h5 读成 numpy 数组（非 PyTorch 场景用）
    """
    with h5py.File(h5_path, "r") as f:
        x1 = f["x1"][:]
        x2 = f["x2"][:]
        x3 = f["x3"][:]
        x4 = f["x4"][:]
        x5 = f["x5"][:]
        y  = f["y"][:]
    return x1, x2, x3, x4, x5, y
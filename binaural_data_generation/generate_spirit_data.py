import os
import glob
import numpy as np
import soundfile as sf
import pysofaconventions as sofa
from scipy.signal import fftconvolve, resample_poly
from tqdm import tqdm

# ============================================================
# Spirit reverberant dataset generator (Anechoic-compatible)
# Output:
#   <ds_name>_<i:06d>.wav
#   <ds_name>_<i:06d>.npz
#
# Labels:
#   aoa_deg: relative azimuth in [0,360) (speaker_world - head_yaw, wrapped)
#   sector_id, sector_angle_norm: derived from aoa_deg (8 sectors)
#   distance_m: physical distance from (x,y)
#   distance_class: 5 classes:
#       0: 0.5m
#       1: 1.0m
#       2: 2.0m
#       3: 3.0m
#       4: other (dist > 3.0m)
# ============================================================

# =========================
# CONFIG
# =========================
SOFA_FILE = "/media/mengh/SharedData/hanyu/DeepEar/TU_Berlin/QU_KEMAR_spirit.sofa"
TIMIT_ROOT = "/media/mengh/SharedData/hanyu/AuralNet/TIMIT"   # TRAIN / TEST
OUT_ROOT  = "/media/mengh/SharedData/hanyu/DeepEar/spirit_dataset"

FS_TARGET = 16000
SEGMENT_SEC = 1.0
N_SECTORS = 8

# 5-class distance labeling (same idea as Auditorium3)
DIST_PROTOS = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
OTHER_THRESH_M = 3.0
OTHER_CLASS_ID = 4

# Optional: if you find all AoA is rotated by a constant (e.g., 90deg), set here.
# Start with 0.0; only change after sanity check.
AZ_OFFSET_DEG = 0.0

DATASET_SPECS = {
    "spirit_test": {"num": 9000, "split": "TEST"},
}

os.makedirs(OUT_ROOT, exist_ok=True)

# =========================
# Spirit setup (from your PDF)
# KEMAR at (0,0); 3 loudspeakers (x,y) in meters
# =========================
SPEAKER_XY = np.array([
    [-1.00,  1.73],  # Spk1
    [ 0.00,  2.00],  # Spk2
    [ 1.00,  1.73],  # Spk3
], dtype=float)

SPEAKER_DIST_M = np.sqrt(np.sum(SPEAKER_XY**2, axis=1))

def wrap_0_360(deg: float) -> float:
    deg = float(deg) % 360.0
    if deg < 0:
        deg += 360.0
    return deg

def wrap_to_180(deg: float) -> float:
    deg = float(deg) % 360.0
    if deg > 180.0:
        deg -= 360.0
    return deg

def aoa_to_sector_label(az_deg: float, n_sectors: int = N_SECTORS):
    az = wrap_0_360(az_deg)
    w = 360.0 / n_sectors
    sector_id = int(np.floor(az / w))
    sector_id = max(0, min(n_sectors - 1, sector_id))
    in_sector = az - sector_id * w
    eps = 1e-3
    norm_ang = (in_sector + eps) / w
    norm_ang = min(norm_ang, 1.0)
    return sector_id, norm_ang

def distance_to_class_index_5(dist_m: float) -> int:
    """
    5-class:
      dist > 3.0m -> class 4 ("other")
      else -> nearest of [0.5, 1, 2, 3] (argmin)
    """
    dist_m = float(dist_m)
    if dist_m > OTHER_THRESH_M:
        return OTHER_CLASS_ID
    return int(np.argmin(np.abs(DIST_PROTOS - dist_m)))

def world_azimuth_from_xy(xy):
    """
    Define world azimuth from x,y:
      az = atan2(y, x) in degrees, wrapped to [0,360)
    NOTE:
      This sets 0° at +x axis. If you want 0° at +y (front),
      you can set AZ_OFFSET_DEG = -90 (or +90) after sanity checks.
    """
    return wrap_0_360(np.degrees(np.arctan2(xy[1], xy[0])) + AZ_OFFSET_DEG)

SPEAKER_AZ_WORLD = np.array([world_azimuth_from_xy(SPEAKER_XY[i]) for i in range(len(SPEAKER_XY))], dtype=float)

# =========================
# 1) Load SOFA BRIR
# =========================
hrtf = sofa.SOFAFile(SOFA_FILE, "r")
IR = hrtf.getDataIR()  # expected (M, 2, S, N)
fs_brir = float(hrtf.getSamplingRate())

if IR.ndim != 4:
    raise ValueError(f"Expect 4D IR (M,2,S,N). Got {IR.shape}")

M, R, S, N = IR.shape
assert R == 2, "Expect 2 receivers (L/R)"
assert S == len(SPEAKER_XY), f"Spirit expects {len(SPEAKER_XY)} sources, got S={S}"
print(f"[SOFA] IR shape={IR.shape}, fs={fs_brir}")

# =========================
# 1b) Head yaw axis extraction (best-effort)
# =========================
def vec_to_az_deg(v):
    v = np.asarray(v).reshape(-1)
    return wrap_0_360(np.degrees(np.arctan2(v[1], v[0])))

def get_head_yaw_axis(hrtf, M):
    """
    Best-effort extraction of head yaw (deg) for each measurement index m.

    Priority:
      1) ListenerView (vector) -> azimuth
      2) ListenerView (deg)
      3) SourcePosition (deg or vector) fallback
      4) fallback linspace(-90,90)
    """
    # 1) ListenerView
    try:
        lv = np.squeeze(np.array(hrtf.getVariableValue("ListenerView")))
        if lv.ndim == 2 and lv.shape[0] == M and lv.shape[1] >= 2:
            if lv.shape[1] >= 3:
                norms = np.linalg.norm(lv[:, :3], axis=1)
                if 0.5 < np.median(norms) < 2.0:
                    yaw = np.array([vec_to_az_deg(lv[m, :3]) for m in range(M)], dtype=float)
                    return yaw, "ListenerView(vector)->az"
            if -360 <= np.nanmin(lv[:, 0]) and np.nanmax(lv[:, 0]) <= 360:
                yaw = np.array([wrap_0_360(a) for a in lv[:, 0].astype(float)], dtype=float)
                return yaw, "ListenerView[:,0](deg)"
    except Exception:
        pass

    # 2) SourcePosition fallback
    try:
        sp = np.squeeze(np.array(hrtf.getVariableValue("SourcePosition")))
        if sp.ndim == 2 and sp.shape[0] == M and sp.shape[1] >= 1:
            if -360 <= np.nanmin(sp[:, 0]) and np.nanmax(sp[:, 0]) <= 360:
                yaw = np.array([wrap_0_360(a) for a in sp[:, 0].astype(float)], dtype=float)
                return yaw, "SourcePosition[:,0](deg)"
            if sp.shape[1] >= 3:
                norms = np.linalg.norm(sp[:, :3], axis=1)
                if 0.5 < np.median(norms) < 2.0:
                    yaw = np.array([vec_to_az_deg(sp[m, :3]) for m in range(M)], dtype=float)
                    return yaw, "SourcePosition(vector)->az"
    except Exception:
        pass

    return (np.linspace(-90.0, 90.0, M, dtype=float) % 360.0), "fallback linspace(-90,90)"

HEAD_YAW_DEG, HEAD_SRC = get_head_yaw_axis(hrtf, M)
print(f"[SOFA] Head yaw axis source: {HEAD_SRC}, shape={HEAD_YAW_DEG.shape}")

# Optional: Only set this if you have verified a known room reference offset.
# Start from 0.0 and change after sanity check if needed.
HEAD_YAW_OFFSET_DEG = 0.0
HEAD_YAW_DEG = (HEAD_YAW_DEG + HEAD_YAW_OFFSET_DEG) % 360.0

# =========================
# 2) Resample BRIR to 16k
# =========================
def resample_brir_4d(IR_4d, fs_in, fs_out):
    if int(fs_in) == int(fs_out):
        return IR_4d
    M0, R0, S0, N0 = IR_4d.shape
    gcd = np.gcd(int(fs_in), int(fs_out))
    up = int(fs_out // gcd)
    down = int(fs_in // gcd)

    out = []
    for m in range(M0):
        rr = []
        for r in range(R0):
            ss = []
            for s in range(S0):
                y = resample_poly(IR_4d[m, r, s, :], up, down)
                ss.append(y)
            rr.append(np.stack(ss, axis=0))
        out.append(np.stack(rr, axis=0))
    return np.stack(out, axis=0)

IR_16k = resample_brir_4d(IR, fs_brir, FS_TARGET)
print(f"[SOFA] Resampled IR -> {FS_TARGET} Hz, shape={IR_16k.shape}")

# =========================
# 3) Collect TIMIT wavs
# =========================
def collect_timit_wavs(split):
    base = os.path.join(TIMIT_ROOT, split)
    wavs = sorted(
        glob.glob(os.path.join(base, "**", "*.WAV"), recursive=True)
        + glob.glob(os.path.join(base, "**", "*.wav"), recursive=True)
    )
    if len(wavs) == 0:
        raise RuntimeError(f"No wavs found in {base}")
    print(f"[TIMIT {split}] {len(wavs)} wavs")
    return wavs

def load_random_clean_segment(wav_paths, length_sec, fs_target):
    while True:
        wav_path = np.random.choice(wav_paths)
        x, fs = sf.read(wav_path)
        if x.ndim > 1:
            x = x[:, 0]
        if fs != fs_target:
            gcd = np.gcd(int(fs), int(fs_target))
            up = int(fs_target // gcd)
            down = int(fs // gcd)
            x = resample_poly(x, up, down)
        T = int(length_sec * fs_target)
        if len(x) >= T + int(0.2 * fs_target):
            start = np.random.randint(0, len(x) - T + 1)
            return x[start:start + T].astype(float)

# =========================
# 4) Relative azimuth helpers
# =========================
def relative_azimuth_deg(speaker_az_world_deg, head_yaw_deg):
    return wrap_0_360(speaker_az_world_deg - head_yaw_deg)

def speakers_in_sector(head_yaw_deg, target_sector):
    idxs = []
    for s_id in range(S):
        rel_az = relative_azimuth_deg(SPEAKER_AZ_WORLD[s_id], head_yaw_deg)
        sid, _ = aoa_to_sector_label(rel_az)
        if sid == target_sector:
            idxs.append(s_id)
    return idxs

# =========================
# 5) Synthesize one sample
# =========================
def synthesize_one_sample(clean_wavs, n_sources=None):
    if n_sources is None:
        n_sources = int(np.random.choice([1, 2, 3]))
    else:
        n_sources = int(n_sources)

    T = int(SEGMENT_SEC * FS_TARGET)
    yL = np.zeros(T, dtype=float)
    yR = np.zeros(T, dtype=float)

    m = np.random.randint(0, M)
    head_yaw = float(HEAD_YAW_DEG[m])

    labels = {
        "num_sources": n_sources,
        "aoa_deg": [],
        "distance_m": [],
        "sector_id": [],
        "sector_angle_norm": [],
        "distance_class": [],
        "speaker_id": [],       # (optional but useful)
        "head_yaw_deg": head_yaw,  # (optional but useful)
        "head_index": int(m),      # (optional but useful)
    }

    # build which sectors are possible under this head yaw
    sector_to_candidates = {}
    for sid in range(N_SECTORS):
        cands = speakers_in_sector(head_yaw, sid)
        if len(cands) > 0:
            sector_to_candidates[sid] = cands

    valid_sectors = sorted(list(sector_to_candidates.keys()))
    if len(valid_sectors) == 0:
        n_sources = 1
        s_id = int(np.random.randint(0, S))
        rel_az = float(relative_azimuth_deg(SPEAKER_AZ_WORLD[s_id], head_yaw))
        sid, norm_ang = aoa_to_sector_label(rel_az)
        chosen = [(s_id, rel_az, sid, float(norm_ang))]
    else:
        n_sources_eff = min(n_sources, len(valid_sectors))
        chosen_sectors = np.random.choice(valid_sectors, size=n_sources_eff, replace=False)

        chosen = []
        used_speakers = set()
        for sid in chosen_sectors:
            cands = sector_to_candidates[int(sid)]
            cands2 = [c for c in cands if c not in used_speakers]
            if len(cands2) == 0:
                cands2 = cands
            s_id = int(np.random.choice(cands2))
            used_speakers.add(s_id)

            rel_az = float(relative_azimuth_deg(SPEAKER_AZ_WORLD[s_id], head_yaw))
            sid2, norm_ang = aoa_to_sector_label(rel_az)
            chosen.append((s_id, rel_az, int(sid2), float(norm_ang)))

        n_sources = len(chosen)
        labels["num_sources"] = int(n_sources)

    for (s_id, rel_az, sid, norm_ang) in chosen:
        x = load_random_clean_segment(clean_wavs, SEGMENT_SEC, FS_TARGET)

        hL = IR_16k[m, 0, s_id, :]
        hR = IR_16k[m, 1, s_id, :]

        yL_full = fftconvolve(x, hL)
        yR_full = fftconvolve(x, hR)

        yL_add = yL_full[:T] if len(yL_full) >= T else np.pad(yL_full, (0, T - len(yL_full)))
        yR_add = yR_full[:T] if len(yR_full) >= T else np.pad(yR_full, (0, T - len(yR_full)))

        yL += yL_add
        yR += yR_add

        dist_m = float(SPEAKER_DIST_M[s_id])
        dist_cls = distance_to_class_index_5(dist_m)

        labels["aoa_deg"].append(float(rel_az))
        labels["distance_m"].append(float(dist_m))
        labels["sector_id"].append(int(sid))
        labels["sector_angle_norm"].append(float(norm_ang))
        labels["distance_class"].append(int(dist_cls))
        labels["speaker_id"].append(int(s_id))

    mx = max(np.max(np.abs(yL)), np.max(np.abs(yR)), 1e-8)
    yL = 0.9 * yL / mx
    yR = 0.9 * yR / mx

    y = np.stack([yL, yR], axis=0)  # (2, T)
    return y, labels

# =========================
# 6) Generate dataset
# =========================
def generate_dataset(ds_name, num_samples, split):
    print(f"\n=== Generating {ds_name} ({num_samples} samples, TIMIT {split}) ===")
    clean_wavs = collect_timit_wavs(split)

    print("\n[Spirit] Speaker geometry:")
    for i in range(S):
        print(f"  Spk{i}: xy={SPEAKER_XY[i].tolist()}, dist={SPEAKER_DIST_M[i]:.3f}m, az_world={SPEAKER_AZ_WORLD[i]:.2f}deg")

    print("\n[Distance classes] protos:", DIST_PROTOS.tolist(), f"other: id={OTHER_CLASS_ID}, thresh>{OTHER_THRESH_M}m")

    out_dir = os.path.join(OUT_ROOT, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    is_test = ds_name.endswith("test") or ds_name.endswith("_test")
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
            n_src = None

        y, labels = synthesize_one_sample(clean_wavs, n_sources=n_src)

        wav_path = os.path.join(out_dir, f"{ds_name}_{i:06d}.wav")
        sf.write(wav_path, y.T, FS_TARGET)

        npz_path = os.path.join(out_dir, f"{ds_name}_{i:06d}.npz")
        np.savez(
            npz_path,
            audio_path=wav_path,
            num_sources=int(labels["num_sources"]),
            aoa_deg=np.array(labels["aoa_deg"], dtype=np.float32),
            distance_m=np.array(labels["distance_m"], dtype=np.float32),
            sector_id=np.array(labels["sector_id"], dtype=np.int64),
            sector_angle_norm=np.array(labels["sector_angle_norm"], dtype=np.float32),
            distance_class=np.array(labels["distance_class"], dtype=np.int64),

            # optional debugging fields (safe to keep)
            speaker_id=np.array(labels["speaker_id"], dtype=np.int64),
            head_yaw_deg=np.float32(labels["head_yaw_deg"]),
            head_index=np.int64(labels["head_index"]),
            dist_protos=np.array(DIST_PROTOS, dtype=np.float32),
            other_thresh_m=np.float32(OTHER_THRESH_M),
            other_class_id=np.int64(OTHER_CLASS_ID),
            az_offset_deg=np.float32(AZ_OFFSET_DEG),
            head_yaw_offset_deg=np.float32(HEAD_YAW_OFFSET_DEG),
        )

    print(f"Finished {ds_name}, saved to {out_dir}")

def main():
    for ds_name, spec in DATASET_SPECS.items():
        generate_dataset(ds_name, spec["num"], spec["split"])
    print("\n✅ All Spirit datasets generated under:", OUT_ROOT)

if __name__ == "__main__":
    main()
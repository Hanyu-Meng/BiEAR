# BiEAR

**BiEAR: A Human Auditory-Inspired Adaptive Binaural Front-end for Multi-Speaker Localisation and Distance Estimation**

This repository contains the official implementation of BiEAR submitted to interspeech2026, including binaural data generation, H5 dataset creation, model definition, training, and evaluation.

---

## Overview

BiEAR uses an adaptive binaural front-end (ERB-based gammatone filterbank with learnable Q) and a joint backend for:

- **Sound presence** per sector  
- **Angle-of-arrival (AoA)** within sectors  
- **Distance** estimation (multi-class)

Training supports both **passive** (precomputed features from H5) and **active** (raw waveform → front-end → backend) modes, with optional cross-correlation (CC) features and fixed or adaptive Q.

---

## Required datasets

Binaural data generation in this repo relies on the **TU Berlin HRIR/BRIR dataset** (KEMAR manikin, SOFA format). You need to obtain the SOFA files and point the scripts to them.

### TU Berlin KEMAR (SOFA)

The code expects SOFA files from the TU Berlin binaural database:

| Script | SOFA file (set `SOFA_FILE` in script) | Description |
|--------|----------------------------------------|-------------|
| `generate_anechoic_data.py` | `QU_KEMAR_anechoic.sofa` | Anechoic HRIRs; multiple distances (e.g. 0.5 m, 1 m, 2 m, 3 m). |
| `generate_auditorium_data.py` | `QU_KEMAR_Auditorium3.sofa` | Room BRIRs (auditorium). |
| `generate_spirit_data.py` | `QU_KEMAR_spirit.sofa` | BRIRs for Spirit configuration. |

**Where to get the data**

- **Anechoic HRIRs (SOFA):** TU Berlin QU KEMAR anechoic data in SOFA format is available from the [SOFA database index](https://sofacoustics.org/data/database/tu-berlin/) (e.g. `qu_kemar_anechoic_0.5m.sofa`, `qu_kemar_anechoic_1m.sofa`, …, or `qu_kemar_anechoic_all.sofa`). You may need to merge or rename files to match the single `QU_KEMAR_anechoic.sofa` path used in the script, or edit the script to load the per-distance files.
- **Room BRIRs (Auditorium, Spirit):** TU Berlin also provides binaural room impulse responses (e.g. via [DepositOnce](https://depositonce.tu-berlin.de)); check the TU Berlin / SOFA pages for Spirit and Auditorium SOFA files. Place the files in a folder (e.g. `TU_Berlin/`) and set `SOFA_FILE` in each script to the full path.

If you use the TU Berlin anechoic KEMAR HRIR data, please cite the dataset as given in [References](#references).

### Speech corpus (for anechoic data generation)

`generate_anechoic_data.py` uses a clean-speech corpus for source signals (e.g. **TIMIT**). Set `TIMIT_ROOT` in the script to the path that contains `TRAIN` and `TEST` (or your own corpus layout). You are responsible for obtaining TIMIT or an equivalent corpus and complying with its license.

### Summary

1. Download the TU Berlin SOFA files (anechoic and, if needed, Auditorium/Spirit) and place them in a directory, e.g. `TU_Berlin/`.
2. In each script under `binaural_data_generation/`, set `SOFA_FILE` to the full path of the corresponding SOFA file (e.g. `.../TU_Berlin/QU_KEMAR_anechoic.sofa`).
3. For anechoic generation, set `TIMIT_ROOT` and `OUT_ROOT` as needed.

---
## Environment

### Requirements

- **Python**: 3.8+
- **PyTorch**: 1.10+ (with CUDA if you want GPU training)
- **Other**: See `requirements.txt` for Python dependencies.

### Setup (Conda, recommended)

```bash
# Create and activate environment
conda create -n biear python=3.10 -y
conda activate biear

# Install PyTorch (CPU or CUDA; adjust for your driver/CUDA version)
# CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# Or with CUDA 11.8:
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Python dependencies
pip install -r requirements.txt
```

### Setup (pip only)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install torch torchvision torchaudio   # add CUDA variant if needed
pip install -r requirements.txt
```

### Main dependencies (from `requirements.txt`)

| Package           | Purpose                    |
|-------------------|----------------------------|
| `torch`           | Model and training         |
| `numpy`, `scipy`  | Numerics and signal        |
| `h5py`            | H5 dataset I/O            |
| `PyYAML`          | Config (`conf/config.yaml`)|
| `tensorboard`     | Training logs              |
| `soundfile`       | Audio I/O                 |
| `librosa`         | Audio utilities            |
| `gammatone`       | Gammatone filterbank       |
| `pysofaconventions` | SOFA HRIR (data generation) |
| `tqdm`            | Progress bars              |

### Development environment (this repo)

The code was developed and tested on this machine using the conda environment **`pytorch_env`**:

- **Python**: 3.11.9  
- **PyTorch**: 2.7.0+cu128 (CUDA 12.8), with `torchaudio` and `torchvision`  
- **Other packages**: See `requirements-pytorch_env.txt` for pinned versions used in that env.

To run with the same environment:

```bash
conda activate pytorch_env
cd /path/to/BiEAR
# Ensure data module is on path if needed (e.g. for training)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/create_h5_data"
python train_biear.py   # or evaluate_biear.py
```

To replicate a similar environment from scratch (with CUDA 12.x):

```bash
conda create -n biear python=3.11 -y
conda activate biear
pip install torch torchvision torchaudio  # or install with CUDA from pytorch.org
pip install -r requirements-pytorch_env.txt
```

---

## Project structure

```
BiEAR/
├── conf/
│   ├── config.yaml              # Main training config
│   ├── config_single_ctrl.yaml
│   └── config_auralnet_deepear.yaml
├── binaural_data_generation/    # Binaural dataset generation
│   ├── generate_anechoic_data.py
│   ├── generate_auditorium_data.py
│   └── generate_spirit_data.py
├── create_h5_data/              # Build H5 datasets from raw data
│   ├── data_save.py             # Dataset classes & loading
│   ├── data_h5_save.py          # H5 writing
│   ├── precompute_h5.py         # Script to run H5 creation
│   └── utils_save.py
├── model_torch.py               # BiEAR model (front-end + backend)
├── train_biear.py               # Training script
├── evaluate_biear.py            # Evaluation script
├── utils.py                     # Gammatone / ERB utilities
├── requirements.txt            # Minimal deps (no versions)
├── requirements-pytorch_env.txt  # Pinned versions from pytorch_env
└── README.md
```

---

## Configuration

Training is driven by **`conf/config.yaml`**. Important fields:

| Key | Description |
|-----|-------------|
| `ROOT` | Path to dataset root (e.g. anechoic train/val H5 or raw data parent). |
| `BATCH_SIZE`, `EPOCHS` | Training schedule. |
| `Active` | `true`: waveform input; `false`: precomputed features from H5. |
| `USE_CC` | Use cross-correlation features. |
| `FIXED_FRONTEND_Q` | `true`: fixed Q; `false`: adaptive Q. |
| `Controller_Mode` | `"dual"` (or single-controller configs). |
| `LOSS_WEIGHT_SOUND` / `LOSS_WEIGHT_AOA` / `LOSS_WEIGHT_DIST` | Loss weights (should sum to 1.0). |
| `RUNS_ROOT` | Parent folder for run directories (checkpoints, TensorBoard, logs). |

Adjust `ROOT` and paths inside the data-generation scripts to match your machine.

---

## Data pipeline

### 1. Binaural data generation

Generate binaural (`.npz` + `.wav`) data using SOFA HRIRs and source signals:

- **Anechoic**: `binaural_data_generation/generate_anechoic_data.py`  
- **Auditorium**: `binaural_data_generation/generate_auditorium_data.py`  
- **Spirit**: `binaural_data_generation/generate_spirit_data.py`  

Edit the script headers to set:

- `SOFA_FILE`: path to SOFA HRIR (e.g. QU_KEMAR_anechoic.sofa)  
- `TIMIT_ROOT` / source corpus paths  
- `OUT_ROOT`: output directory for datasets  

### 2. H5 dataset creation

From a directory of `.npz`/`.wav` samples, build H5 files for training/validation:

```bash
cd create_h5_data
# Edit precompute_h5.py: set ROOT and dataset_dir / h5_path
python precompute_h5.py
```

This uses `data_save.py` and `data_h5_save.py` to produce H5 files with `x1`, `x2`, `x3`, … and labels `y`.

### 3. Training

Point `conf/config.yaml` at your H5 dataset root (or at the directory containing train/val H5 paths used by your data loader). Then:

```bash
# From repo root; ensure create_h5_data is on PYTHONPATH if your data module lives there
export PYTHONPATH="${PYTHONPATH}:$(pwd)/create_h5_data"

python train_biear.py
```

Checkpoints and TensorBoard logs are written under `RUNS_ROOT` as specified in the config. For **active** (waveform) training, the data module must provide `DeepEarH5Dataset_Active` (waveform samples); for **passive** training, `DeepEarH5Dataset` (precomputed features) is used.

### 4. Evaluation

Set `CHECKPOINT_PATH` in `evaluate_biear.py` to a trained checkpoint (or the run directory that contains `meta/settings.json`). Then:

```bash
python evaluate_biear.py
```

Evaluation uses the same config as training when loaded from the checkpoint’s `settings.json` (recommended).

---

## Quick start (after environment is ready)

1. **Install environment** (see [Environment](#environment)).  
2. **Obtain required datasets** (see [Required datasets](#required-datasets)): TU Berlin SOFA files and, for anechoic generation, a speech corpus (e.g. TIMIT).  
3. **Generate or obtain binaural data** (e.g. anechoic), then **build H5** via `create_h5_data`.  
4. Set **`ROOT`** and **`RUNS_ROOT`** in `conf/config.yaml`.  
5. Run **`python train_biear.py`** (with `PYTHONPATH` including `create_h5_data` if needed).  
6. Run **`python evaluate_biear.py`** with the desired **`CHECKPOINT_PATH`**.

---

## References

[1] H. Wierstorf, M. Geier, and S. Spors, “A free database of head-related impulse response measurements in the horizontal plane with multiple distances,” in *130th Convention of the Audio Engineering Society*, 2011. SOFA format: [Zenodo 55418](https://doi.org/10.5281/zenodo.55418); [sofacoustics.org TU Berlin](https://sofacoustics.org/data/database/tu-berlin/).

---

## License

See the repository license file (if present). For use of TIMIT, SOFA, or other external data, comply with their respective licenses.

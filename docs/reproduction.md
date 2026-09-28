# Setup and reproduction

[Back to BiEAR](../README.md) · [Configuration](configuration.md) · [Reported results](results.md)

This guide describes the checked-in research code. Review the release notes below before preparing a full experiment.

## Current release notes

The public snapshot has the following gaps:

| Area | Current state |
| :--- | :--- |
| Helper modules | `data.py` imports `utils`, and both trainers import `visualize_q`. Neither `utils.py` nor `visualize_q.py` is included. Training/evaluation cannot run unchanged until the required helpers are restored. `create_h5_data/utils_save.py` is a separate preparation utility. |
| H5 preparation | The current writer creates waveform datasets (`x1`, `x2`, `x3`, `y`). The passive loader additionally requires phase features (`x4`, `x5`), which this writer does not produce. |
| Distance labels | The H5 label builder reserves index 0 for “no source” and shifts the four anechoic distance classes to indices 1–4. The room generators can emit class 4 for “other,” which would be shifted to index 5 in a five-element vector. This needs reconciliation before preparing those room examples. The presentation instead describes five physical distance classes. |
| Paper settings | The checked-in defaults and training logic differ from the presentation. See the [configuration reference](configuration.md#paper-and-code-settings). |
| Baseline/transfer scripts | A baseline configuration and AuralNet model builder are included, but dedicated baseline-training and room fine-tuning entry points are not included. |
| Data and weights | TIMIT, SOFA files, prepared H5 datasets and pretrained checkpoints are not bundled. |
| Environment | `requirements.txt` lists direct dependencies from the source imports. It is not an experimentally validated lockfile. The previously mentioned `requirements-pytorch_env.txt` is not included. |

The documentation refresh does not change model behaviour or resolve these research-code gaps. Commands below describe the workflow **after the corresponding prerequisites are satisfied**.

## 1. Environment

Use a separate Python environment. Python 3.11 is an example matching the development version noted in the original README; compatibility with every dependency version has not been established.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the appropriate PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/), then install the remaining direct dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements include numerical/audio libraries, H5 and SOFA readers, TensorBoard and Matplotlib. No separate `torchvision` or `torchaudio` import appears in the checked-in source.

## 2. Source data

Obtain the speech corpus and measured binaural impulse responses separately. Follow the licences provided by the respective data owners.

| Input | Expected local name/layout | Used by |
| :--- | :--- | :--- |
| TIMIT | Directory containing `TRAIN/` and `TEST/` | All three generators |
| Anechoic KEMAR HRIRs | `QU_KEMAR_anechoic.sofa` | `generate_anechoic_data.py` |
| Lecture-hall BRIRs | `QU_KEMAR_Auditorium3.sofa` | `generate_auditorium_data.py` |
| Meeting-room BRIRs | `QU_KEMAR_spirit.sofa` | `generate_spirit_data.py` |

Starting points: [TU Berlin SOFA database](https://sofacoustics.org/data/database/tu-berlin/), [anechoic HRIR dataset](https://doi.org/10.5281/zenodo.55418) and [TIMIT catalogue](https://catalog.ldc.upenn.edu/LDC93S1). Downloaded filenames can differ from the paths expected by the scripts; inspect the SOFA measurement layout and update the file paths accordingly.

Set `SOFA_FILE`, `TIMIT_ROOT` and `OUT_ROOT` in the relevant file under [`binaural_data_generation/`](../binaural_data_generation/). Review `DATASET_SPECS` before execution: the anechoic default requests 72,000 training examples and 9,000 examples for each validation/test split.

```bash
# Run the generator for the environment you need, after setting local paths.
python binaural_data_generation/generate_anechoic_data.py
```

Each generated sample includes a binaural waveform and `.npz` metadata. Metadata contains an `audio_path`; moving the dataset may require updating those paths.

## 3. Prepare H5 files

[`create_h5_data/precompute_h5.py`](../create_h5_data/precompute_h5.py) calls the waveform H5 writer. Set `ROOT`, `dataset_dir` and `h5_path` for each intended split. Its checked-in active call targets `spirit_test2`, rather than the anechoic training split. Address the distance-label issue above before preparing room data.

```bash
cd create_h5_data
python precompute_h5.py
cd ..
```

### Active waveform schema

Expected by `DeepEarH5Dataset_Active` in [`data.py`](../data.py):

| H5 key | Shape | Content |
| :--- | :--- | :--- |
| `x1` | `(N, 16000)` for 1 s at 16 kHz | Left waveform |
| `x2` | `(N, 16000)` for 1 s at 16 kHz | Right waveform |
| `x3` | `(N, 100)` | Cross-correlation features |
| `y` | `(N, 56)` | Eight sectors × `[activity, normalised azimuth, five distance entries]` |

The waveform inputs and features should be floating-point arrays. The generator uses one source per occupied sector.

### Passive feature schema

`DeepEarH5Dataset` expects `x1`, `x2`, `x3`, `x4`, `x5` and `y`. The first two arrays contain left/right magnitude features, and `x4`/`x5` contain phase features. Their feature dimensions are 19 time steps × 100 bands per example; `x3` and `y` retain the formats above. The checked-in waveform writer does **not** create this passive dataset.

## 4. Configure and train

Once the missing helper modules are restored and compatible H5 files exist, edit `ROOT` and `RUNS_ROOT` in the matching YAML file:

| Variant | Configuration | Entry point |
| :--- | :--- | :--- |
| Independent ear controllers | [`conf/config.yaml`](../conf/config.yaml) | `python train_biear.py` |
| One shared controller | [`conf/config_single_ctrl.yaml`](../conf/config_single_ctrl.yaml) | `python train_biear_single_ctrl.py` |

Both entry points read a fixed YAML path relative to their own file. They do not expose a `--config` flag. Changing `Controller_Mode` alone does not switch training entry points.

For the active dual-controller trainer, `ROOT` must contain:

```text
anechoic_train_active_wav.h5
anechoic_val_active_wav.h5
anechoic_test1_active_wav.h5
```

For passive mode, the dual-controller trainer instead looks for:

```text
anechoic_train_gt_group_phase.h5
anechoic_val_gt_group_phase.h5
anechoic_test2_gt_group_phase.h5
```

Run the chosen trainer from the repository root. The trainers execute work at module level, so importing a training file is not a safe way to inspect its API.

### Run outputs

```text
RUNS_ROOT/<experiment>/
├── checkpoints/       # best.pth, last.pth and optional epoch checkpoints
├── meta/settings.json # Recorded model and training settings
├── logs_json/         # History and evaluation metrics
├── tb/                # TensorBoard events
└── q_vis/             # Q visualisations, once the missing helper is restored
```

Keep `meta/settings.json` with its checkpoint so evaluation can reconstruct the model settings.

```bash
tensorboard --logdir runs_deepear_exps
```

Use your configured `RUNS_ROOT` if it differs from this default.

## 5. Evaluate

In [`evaluate_biear.py`](../evaluate_biear.py), set:

- `CHECKPOINT_PATH` to an actual checkpoint file, such as `<run>/checkpoints/best.pth`.
- `ROOT` and `test_h5` in the active/passive data-path block to the desired test split.

The evaluator looks for `meta/settings.json` around the checkpoint path. The default test split is `anechoic_test2`. A run-directory path alone is not a checkpoint file.

```bash
python evaluate_biear.py
```

The script reports overall and per-speaker-count metrics. Check label conventions, metric definitions and checkpoint settings against the paper before making numerical comparisons.

## 6. Inspect adaptation

[`plots/`](../plots/) contains scripts for Q trajectories, filter responses and feature visualisations. Update their checkpoint/data paths and model settings before use. For example, once the data imports are available:

```bash
PYTHONPATH=. python plots/plot_Fig3_b.py
```

These scripts read trained weights; the PNGs used in the README are supplied presentation assets rather than figures regenerated by these commands.

## Code licence

No code licence file is currently included. Third-party datasets retain their own terms. The publication's licence and the software's licence are separate.

# Configuration

[Back to BiEAR](../README.md) · [Setup and reproduction](reproduction.md)

## Entry points and variants

| Variant | Entry point | Configuration |
| :--- | :--- | :--- |
| Dual controllers | `train_biear.py` | `conf/config.yaml` |
| Shared controller | `train_biear_single_ctrl.py` | `conf/config_single_ctrl.yaml` |
| Baseline reference | No dedicated trainer included | `conf/config_auralnet_deepear.yaml` |

The main configuration selects active waveform input, dual controllers and relative Q modulation. The shared-controller configuration currently selects absolute Q modulation.

## Main controls

| Key | Meaning |
| :--- | :--- |
| `ROOT` | Directory containing the H5 files named in the trainer |
| `RUNS_ROOT` | Parent directory for checkpoints, settings and logs |
| `Active` | `true`: raw waveforms and an in-model front-end; `false`: precomputed magnitude/phase features |
| `USE_CC` | Include the cross-correlation representation |
| `FIXED_FRONTEND_Q` | Hold front-end Q at its baseline value in active mode |
| `FREEZE_Q_CONTROLLER_ONLY` | Freeze controller weights; input-dependent Q can still vary |
| `Controller_Mode` | Descriptive variant setting; use the corresponding training script |
| `DELTAQ_MODE` | `absolute` or `relative` Q modulation |
| `DELTAQ_BASE` | Overall scale of the frequency-dependent modulation profile |
| `DELTAQ_LOW_FACTOR`, `DELTAQ_HIGH_FACTOR` | Low/high-frequency factors used to construct that profile |
| `LR_FB`, `LR_BACKEND` | Learning rates for the front-end and prediction network |
| `REG_Q_W`, `REG_SMOOTH_W` | Q regularisation coefficients used by the trainer |
| `LOSS_WEIGHT_SOUND`, `LOSS_WEIGHT_AOA`, `LOSS_WEIGHT_DIST` | Multipliers for the three task losses; the code sums the weighted terms without normalising them |
| `BATCH_SIZE`, `EPOCHS` | Batch size and configured epoch count |
| `WEIGHT_DECAY`, `GRAD_CLIP_NORM` | Optimiser weight decay and gradient clipping |
| `SAVE_EVERY_EPOCH` | Save an additional checkpoint each epoch |
| `COMMENTS` | Short suffix included in the run name |

`ALPHA` is passed to the front-end constructor as `fb_alpha`. Inspect the selected implementation before treating it as a controller-history smoothing setting.

## Q modulation

For the controller output `delta` bounded by `tanh`, the implementation uses:

```text
absolute: Q = Q0 + deltaQ_vec * delta
relative: Q = Q0 * (1 + deltaQ_vec * delta)
```

`deltaQ_vec` varies across frequency bands. The adaptive front-end clamps Q to `[0.05, 30.0]`. `DELTAQ_HIGH_FACTOR: 5` is a factor of five in the modulation profile, not 5% or 60%.

## Paper and code settings

The presentation describes the reported experiments. The checked-in configuration is a research snapshot and is not a frozen reproduction recipe.

| Setting | Conference presentation | Current dual-controller code/default |
| :--- | :--- | :--- |
| Epoch schedule | Up to 100 epochs; early stopping patience 10 | `EPOCHS: 150`; full epoch loop with a learning-rate scheduler, without an early-stopping break |
| Learning rate | Adam, `1e-4` | Front-end `5e-5`; back-end `1e-4` |
| Task weights (activity / azimuth / distance) | `0.25 / 0.45 / 0.35` | `0.20 / 0.45 / 0.35` |
| Azimuth loss | Mean squared error | `SmoothL1Loss(beta=0.02)` |
| Loss masking | Azimuth/distance on active sectors | Task losses operate over all sectors |
| Best checkpoint | Lowest validation loss | Lexicographic validation metrics: higher activity accuracy, then lower azimuth error, then higher distance accuracy |
| Distance labels | Five physical distance classes, including “other” | H5 label builder reserves index 0 for no source; see [release notes](reproduction.md#current-release-notes) |

Reconcile these differences with the original experiment settings before reporting reproduced paper results. The presentation's loss weights need not sum to one: the objective is a weighted sum.

## Local paths

The original research paths remain in the configuration and scripts to preserve behaviour. Set paths for your machine in the YAML files, data generators, H5 preparation script, evaluator and plotting scripts before running them. Changing a training YAML does not automatically update the evaluator's test-data paths.

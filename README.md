# SPY-EMA-V2 — chart-only, five-trading-session direction AI

This is a **new experiment**, separate from [SPY v1](https://github.com/jamelof23/SPY). It deliberately **does not use pretrained ImageNet weights or predict next-candle body size**.

## Prediction definition

Given the last **42 completed daily SPY candles**, EMA(8), EMA(21), and daily volume as pixels in a chart, predict:

- **UP (1):** `Close[t+5 trading sessions] > Close[t]`
- **DOWN (0):** `Close[t+5 trading sessions] < Close[t]`
- Exact ties are omitted from binary supervision.

The model receives **image pixels only** (cropped price/EMA and volume panels), not numerical OHLC/EMA features, future price values, or trade annotations. The five-session future closing price is used **only to create the target during historical training**.

This is not the same as next-candle green/red, an EMA rejection entry, or an options profit forecast. A further strategy model would need entries/exits and often five-minute data.

## Split policy and class balance

| Split | As-of dates | Charts | UP after five sessions | DOWN |
|---|---|---:|---:|---:|
| Training | 2020–2024 | **1,253** | 752 | 501 |
| Validation/development | 2025 | **245** | 155 | 90 |
| **Final holdout** | 2026 | **LOCKED** | Not evaluated | Not evaluated |

Five-session labels crossing 2024→2025 or 2025→2026 are *purged*, so neither supervised split uses outcomes from another split. Adjacent 42-chart windows and five-session returns overlap heavily; these records aren't independent observations.

A naive always-UP strategy reaches **63.3% accuracy** on the 2025 development labels. The v2 model must improve on that benchmark—and prove it can recognize DOWN moves in bearish rallies—before it merits additional evaluation.

## Bearish EMA rebound diagnostic

For evaluating, not for model input, each known-as-of chart is tagged when: EMA8 is below EMA21; EMA21 is falling; current close below EMA21; at least two of the most recent three candles are green and the price is rebounding; and the observed price range came within 0.5 past-only ATR of either EMA. This defines **72 training setup-days (~25 separated episodes)** and only **9 development setup-days in 2025**, which is statistically too little to justify an 80% accuracy claim on that subgroup. The tag is not an automatic assertion that a short entry is executable.

## Train with a GPU in Google Colab

1. The notebook, training code and audited labels are in this repository.
2. The [v2 release](https://github.com/jamelof23/SPY-EMA-V2/releases/tag/spy42-v2-images-v1) hosts the supplied archive of 1,498 pre-2026 images. The notebook verifies SHA-256 `030d0d8bc9306a80b57be0b66f9f90a0dc41c716514cc1b29964db800158095f` and stops if this exact asset is unavailable.
3. Open [**SPY v2 Colab notebook from GitHub**](https://colab.research.google.com/github/jamelof23/SPY-EMA-V2/blob/main/notebooks/SPY_V2_5Day_Direction_Colab.ipynb).
4. In Colab select **Runtime → Change runtime type → T4 GPU** (if available), then **Run all**. The notebook clones this repository, downloads images, runs a short GPU benchmark, then trains the CNN.
5. Authorize **Google Drive mount** if prompted, so **every completed epoch** is saved under `MyDrive/SPY-EMA-V2/model_seed42`. The same configuration resumes from the last checkpoint after a disconnect. The drive-mount permission is the only interactive consent needed if the repository is public.
6. Read the printed **2025 validation report**, including overall/balanced accuracy, separate DOWN/UP precision and recall, confusion matrix, baseline comparisons, bear/EMA setup diagnostics, and model parameters. **2026 final-test outcomes remain untouched.**

The Colab notebook contains no GitHub PAT and does not upload your Drive contents or Colab files to GitHub. To publish model weights back to GitHub later, use an explicitly authorized upload route (not silently created credentials).

## Files

- `notebooks/SPY_V2_5Day_Direction_Colab.ipynb` — runnable GPU notebook with observable training output
- `src/build_labels.py` — independently regenerates 2020–2025 five-session labels from the included pre-2026 source CSV
- `src/prepare_data.py` — downloads and verifies chart ZIP; extracts/resizes only approved training and development images
- `src/build_release_images.py` — republishes 1,498 full-resolution PNGs as a GitHub Release asset
- `src/train_v2.py` — small **584,273-parameter CNN, 100% trainable**, binary direction-only target; automatic epoch checkpoints and development metrics
- `data/source/SPY_daily_2019_2025.csv` — original OHLCV source clipped at December 31, 2025, to reproduce the **labels**; not input features
- `data/labels/*.csv` — audited label CSVs and as-of-only EMA setup flags; no 2026 outcomes
- `.github/workflows/publish-images.yml` — GitHub hosted runner to publish the full-resolution pre-2026 images
- `examples/*.png` — historical EMA context example charts

## Reproducibility / safeguards

- Preprocessed CNN images are 512×224 RGB pixels, cropped from the original 1920×1080 chart's price and volume panels.
- The CNN starts from **random weights**; every layer is trained; binary cross-entropy is class-balanced from *training years only*. No ImageNet features, no body quantile head.
- Best checkpoint is selected by 2025 balanced accuracy with log loss as tie breaker. Early stop has patience 7 and maximum 30 epochs.
- The standard 0.5 classification threshold and confidence thresholds (0.55/0.60/0.65) are specified before viewing the new validation results; report each signal count separately.
- Comparisons include always-UP, always-DOWN and EMA8-vs-EMA21 baselines (baseline OHLC information is not sent to the CNN).
- The 2026 final test will be run **once** only after model and trading rules are frozen; it is neither downloaded nor evaluated by this notebook.
- The actual measured 2-batch CPU wiring benchmark took **0.28 seconds on this execution environment**, implying approximately 22 seconds/epoch for the tiny CNN at that batch size. This is only a rough benchmark and should **not** be mistaken for a measured Colab T4 training duration; disk, worker, compute, and batch effects differ.

The notebook and code are intended for a public repository analogous to the existing public SPY repo. Do not put a personal token into a notebook or commit credentials.

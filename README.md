# ShiftQ

A notebook research prototype for MLX on Apple Silicon. All ML code is in
`ShiftQ_Research.ipynb`.

## Research question

Can sensitivity-based bitwidths and LoRA ranks improve perplexity under the
storage budget of uniform 4-bit/rank-16 LoRA? ShiftQ is a heuristic, not a
novelty or performance claim. The MLX baseline uses affine quantization, not NF4.

## Setup

Use Python 3.12 on an Apple Silicon Mac:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab ShiftQ_Research.ipynb
```

In VS Code, select `.venv/bin/python` as the notebook kernel.

## How to run

Run all cells in order. The notebook is currently configured for Qwen3-1.7B,
256-token contexts, two seeds, all comparisons and 150 updates per training run.
Model and dataset downloads require internet.

For a quick smoke test, set `FAST_MODE=True`, `MODEL_ID=SMOKE_MODEL` and
`TRAIN_STEPS=20`. Full runs require `ALLOW_FULL_RUN=True`. 4B is manual.
Each run saves metrics, source-row hashes, token sequences, allocations and
adapters under `results/<run>/`, with plots under `figures/<run>/`.

## Main experiments

- Unquantized reference and supported uniform 3/4/6/8-bit quantization.
- Uniform, bit-adaptive, rank-adaptive and joint LoRA allocation.
- Random/fixed allocation controls, rank ablations and calibration shift.

Stored tensor bytes include quantization metadata and FP32 adapters. Runtime
MLX memory is reported separately. Timing measures forward passes, not decoding.

## Results

The user-executed 1.7B study completed 25 configurations, including 20 training
runs with 150 updates. Test PPL is the mean over seeds 42 and 43 on the same
64 WikiText sequences (16,384 prediction tokens).

| Method | Test PPL | Tensor MB |
|---|---:|---:|
| Uniform 4-bit/rank-16 | 14.656 | 1037.73 |
| ShiftQ | 14.287 | 1037.73 |
| Random placement | 14.287 | 1037.73 |
| Fixed placement | 14.275 | 1037.73 |
| Uniform 4-bit/rank-4 | 14.278 | 985.43 |

ShiftQ lowers PPL by 2.52% versus rank-16, but matches random placement closely.
Rank-4 has slightly lower mean PPL with 5.04% less storage than ShiftQ.
These two seeds do not establish an advantage from sensitivity-aware allocation.

The bit-only policy returned uniform 4-bit; calibration shift changed 6/196 pairs.
All fine-tuned methods worsened AG News PPL versus the unquantized reference.
Later ablations show roughly 1.02 GB extra active MLX memory; cross-phase peak
memory comparisons need a fresh-kernel check before drawing conclusions.

Next: verify memory in isolated runs, then test validation-selected checkpoints
with more seeds and a larger held-out set. Keep the current test results separate
from any future tuning. Detailed review: `results/20261018T211742253834Z/review.json`.

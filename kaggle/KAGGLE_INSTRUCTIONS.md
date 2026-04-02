# Parameter Golf — Kaggle Setup Instructions

This notebook is made to run train_gpt.py on Kaggle. It has some modifications made to original script to make it compatible with older GPUs like T4 and P100. I tested only on 2 T4s. 

The test run with following parameters for the test run:

- TRAIN_SHARDS = 1
- NUM_GPUS = 2
- MAX_WALLCLOCK = 3600
- ITERATIONS = 100

It took around 43 minutes to finish, with around 13 of them dedicated to downloading data, warm up and step 0. Don't worry if the code won't output anything for 10 minutes after doing warm up.

Average time per training step was around 7 seconds. Note, MAX_WALLCLOCK is set to 1 hour by default, so if you are planing to do the longer training be sure to change it.

If you found some bugs or have some suggestions, please let me know. I used AI to code it.

## What You Need

Two files in the `kaggle/` folder:
- **`train_gpt_kaggle.py`** — The training script (adapted from `train_gpt.py`)
- **`train_gpt_kaggle.ipynb`** — The launcher notebook

---

## Step-by-Step Setup

### 1. Create a Kaggle Dataset with the training script

Kaggle notebooks can't directly upload `.py` files, so we store it as a Dataset:

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → click **New Dataset**
2. Name it something like `parameter-golf-scripts`
3. Upload **`train_gpt_kaggle.py`**
4. Click **Create**

Note the dataset URL — it will be `kaggle.com/datasets/<your-username>/parameter-golf-scripts`. You'll need the slug `your-username/parameter-golf-scripts`.

### 2. Upload the notebook to Kaggle

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → click **New Notebook**
2. Click **File → Import Notebook** → upload **`train_gpt_kaggle.ipynb`**

### 3. Attach the dataset

1. In the notebook sidebar, click **Add Data** (or go to Settings → Data)
2. Search for your dataset (`parameter-golf-scripts`)
3. Add it — it will be mounted at `/kaggle/input/parameter-golf-scripts/`

### 4. Select accelerator

1. Go to **Settings** → **Accelerator**
2. Select **GPU T4 x2** (recommended) or **P100** (single GPU)

### 5. Configure and run

1. Open the **first code cell** (CONFIGURATION)
2. Set `KAGGLE_DATASET_SLUG` to your actual dataset slug (e.g., `"johndoe/parameter-golf-scripts"`)
3. Adjust `TRAIN_SHARDS`, `MAX_WALLCLOCK`, `ITERATIONS` as desired
4. Click **Run All**

---

## Configuration Guide

| Variable | Default | Description |
|----------|---------|-------------|
| `KAGGLE_DATASET_SLUG` | — | Your dataset slug: `username/dataset-name` |
| `TRAIN_SHARDS` | 20 | Training data shards (each ~200MB, 100M tokens). Range: 1–80 |
| `NUM_GPUS` | 2 | GPUs to use. Set to 1 for P100 or single T4 |
| `MAX_WALLCLOCK` | 3600 | Max training time in seconds |
| `ITERATIONS` | 20000 | Max training steps (will stop earlier if wallclock is hit) |

**Disk space note:** Each shard is ~200MB. With 20 shards + validation + model files, you'll use ~5GB of Kaggle's ~20GB disk. You can safely go up to 60 shards.

---

## Changes from Original `train_gpt.py`

Only **5 areas** were changed (across ~15 lines), all marked with `# KAGGLE:` comments. To revert for 8×H100 runs, just restore these lines.

### 1. Default data paths (lines 48, 51)
```diff
-data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
+data_path = os.environ.get("DATA_PATH", "/kaggle/working/data/datasets/fineweb10B_sp1024")  # KAGGLE

-tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
+tokenizer_path = os.environ.get("TOKENIZER_PATH", "/kaggle/working/data/tokenizers/fineweb_1024_bpe.model")  # KAGGLE
```
**Why:** Kaggle's working directory is `/kaggle/working/`. The notebook downloads data there.

### 2. Max wallclock time (line 66)
```diff
-max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
+max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 3600.0))  # KAGGLE
```
**Why:** T4 GPUs are ~8× slower than H100s. 600 seconds (10 min) is too short to get meaningful training on T4. Default increased to 1 hour.

### 3. Auto-detect compute dtype: fp16 on T4, bf16 on Ampere+ (multiple lines)
```diff
+_COMPUTE_DTYPE = torch.bfloat16  # overridden in main() for SM < 8.0
 ...
+_COMPUTE_DTYPE = torch.bfloat16 if _cc >= (8, 0) else torch.float16
```
All `torch.bfloat16` references (autocast, model init, Muon optimizer) now use `_COMPUTE_DTYPE`.

**Why:** T4 (SM 7.5) does not support native bf16 compilation in `torch.compile`. The compiler falls back to fp32, doubling memory usage and causing OOM. Using fp16 on T4 gives native compile support and correct memory usage.

### 4. Auto-detect grad accumulation based on VRAM (lines 758-767)
```diff
-if 8 % world_size != 0:
-    raise ValueError(...)
-grad_accum_steps = 8 // world_size
+_grad_accum_total = 8 if _vram_gb >= 40 else 32  # H100: 8, T4: 32
+grad_accum_steps = _grad_accum_total // world_size
```
**Why:** The original uses 8 total micro-batches (designed for H100 80GB). Each micro-batch on T4 x2 was 64 sequences — too large for 15GB VRAM. With 32 total micro-batches, each is only 8 sequences, fitting easily. Override with `GRAD_ACCUM_TOTAL` env var.

### 5. SDP backend auto-detection (lines 779-786)
```diff
+_cc = torch.cuda.get_device_capability(device)
+_has_flash = _cc >= (8, 0)
 enable_cudnn_sdp(False)
-enable_flash_sdp(True)
-enable_mem_efficient_sdp(False)
-enable_math_sdp(False)
+enable_flash_sdp(_has_flash)
+enable_mem_efficient_sdp(not _has_flash)
+enable_math_sdp(not _has_flash)
```
**Why:** Flash Attention requires SM 8.0+ (Ampere/Hopper GPUs like A100, H100). T4 (SM 7.5) needs the `mem_efficient` or `math` attention backends instead.

---

## Expected Output

After training completes, you'll find in `/kaggle/working/`:
- `final_model.pt` — Raw model checkpoint
- `final_model.int8.ptz` — Quantized + zlib-compressed model (for submission)
- `logs/<run_id>.txt` — Full training log

The last cells show training metrics and file sizes.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: train_gpt_kaggle.py` | Check `KAGGLE_DATASET_SLUG` matches your dataset. Files are at `/kaggle/input/datasets/<slug>/` |
| `No files found for pattern: ...fineweb_train_*.bin` | The data download cell didn't complete. Re-run Cell 3 |
| `GRAD_ACCUM_TOTAL=X must be divisible by WORLD_SIZE=Y` | Set `GRAD_ACCUM_TOTAL` to a multiple of `NUM_GPUS` (e.g., 32 for 2 GPUs) |
| Out of memory | Increase `GRAD_ACCUM_TOTAL` (e.g., `GRAD_ACCUM_TOTAL=64`) in Cell 4 env_overrides |
| Training too slow | Reduce `ITERATIONS` or `TRAIN_SHARDS` for faster experiments |

## P100 Notes

P100 works out of the box with `NUM_GPUS = 1`. The script auto-detects SM 6.0 and uses fp16 + mem_efficient attention.

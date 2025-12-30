# 🔄 Resume Training Guide

This document explains how to properly resume training after interruptions (crashes, OOM errors, manual stops).

## 📋 Table of Contents

- [How Resume Works](#how-resume-works)
- [Common Scenarios](#common-scenarios)
- [Configuration](#configuration)
- [Viewing History Without Training](#viewing-history-without-training)
- [Troubleshooting](#troubleshooting)

---

## How Resume Works

The resume system ensures consistency across interrupted and resumed training sessions:

### 1. **Experiment Directory Management**

- **Fresh Training**: Creates a new timestamped directory (`YYYYMMDD_HHMMSS_v1`)
- **Resume Mode**: Reuses the existing experiment directory (no new folder created)

### 2. **Checkpoint Recovery**

- Automatically searches for `best_model.pth` in the experiment's checkpoints directory
- If missing (e.g., after OOM crash), creates it from the most recent available checkpoint:
  - Emergency checkpoints: `emergency_checkpoint_oom_epoch_XXX.pth`
  - Periodic checkpoints: `checkpoint_epoch_XXX.pth`

### 3. **History Continuity**

- Loads previous `history.json` containing all metrics from past epochs
- Continues appending new metrics to the same history
- Validates history structure before merging

### 4. **State Restoration**

- Model weights
- Optimizer state (momentum, adaptive learning rates)
- Scheduler state (learning rate progression)
- Training epoch counter

---

## Common Scenarios

### Scenario 1: OOM Crash During Training

**What Happens:**

1. Training crashes with CUDA Out of Memory error
2. Emergency checkpoint saved: `emergency_checkpoint_oom_epoch_X.pth`
3. History saved to `history.json`

**How to Resume:**

#### If Notebook is Still Open:

```python
# In Configuration cell, modify:
config["batch_size"] = 4  # Reduce from 8
config["resume_from_checkpoint"] = True
# resume_experiment already points to current exp_dir
```

Then re-run **only** these cells:

- Cell: Configuration (with modified batch_size)
- Cell: Training Loop

⚠️ **DO NOT re-run** "Create Output Directories" - it would load the existing experiment correctly.

#### If Notebook was Closed:

```python
# In Configuration cell:
config["batch_size"] = 4  # Reduce
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "latest"  # or specific timestamp
```

Then run all cells from the beginning. The system will:

- Load the most recent experiment
- Find and use the emergency checkpoint
- Create `best_model.pth` if missing
- Continue training from where it stopped

---

### Scenario 2: Manual Stop (Ctrl+C)

**What Happens:**

1. `KeyboardInterrupt` caught by training loop
2. Final state saved to `history.json`
3. Last periodic checkpoint available

**How to Resume:**

```python
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "latest"
```

Run all cells normally.

---

### Scenario 3: View History Without Training

**Use Case:** You want to plot training curves without starting a new training session.

**Steps:**

1. Set configuration:

```python
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "latest"  # or specific experiment
```

2. Set control variable:

```python
SKIP_TRAINING = True  # Skip the training loop
```

3. Run all cells (Run All)

The notebook will:

- Load the existing experiment
- Skip the training loop
- Load history from `history.json`
- Define all necessary variables (`history`, `best_info`, `best_epoch`, `best_val_loss`)
- Continue to plotting and inference cells normally

---

### Scenario 4: Resume Specific Experiment

**Use Case:** You have multiple experiments and want to continue a specific one.

```python
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "20251230_153045_v1"  # Specific timestamp
```

The system will:

- Load that exact experiment directory
- Resume from its latest checkpoint
- Continue its history

---

## Configuration

### Resume Parameters

```python
config = {
    # Resume Training
    "resume_from_checkpoint": False,  # True to resume, False for fresh start
    "resume_experiment": "latest",     # "latest" or specific timestamp

    # Other configs...
}
```

### Control Variable

```python
# Set after config cell, NOT inside config dict
SKIP_TRAINING = False  # Set to True to skip training and only load results
```

This variable allows you to:

- Run all cells without executing the training loop
- Load existing history and best model info
- Access plotting and inference cells normally
- Avoid undefined variable errors when skipping training

### Parameter Details

| Parameter                | Type   | Description                                                                                                            |
| ------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| `resume_from_checkpoint` | `bool` | Enable/disable resume mode                                                                                             |
| `resume_experiment`      | `str`  | Which experiment to resume:<br>- `"latest"`: Most recent by timestamp<br>- `"20251230_153045_v1"`: Specific experiment |
| `SKIP_TRAINING`          | `bool` | Skip training loop (loads existing results instead)                                                                    |

---

## Viewing History Without Training

The notebooks support viewing results without running training via the `SKIP_TRAINING` control variable.

### How It Works

```python
# Configuration cell
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "latest"

# Control variable cell (separate from config)
SKIP_TRAINING = True  # Enable skip mode
```

Then **Run All**. The notebook will:

1. Load the existing experiment directory
2. Skip the `run_training()` call
3. Load `history.json` instead
4. Extract best epoch/loss from history
5. Define all variables needed for plotting/inference
6. Continue normally to visualization cells

### Benefits

✅ **No undefined variables** - All variables are properly initialized
✅ **Run All compatible** - No need to manually skip cells
✅ **Consistent behavior** - Same variables whether training or loading

### Example Output

```
⏭️  SKIP_TRAINING = True: Loading existing results
================================================================================

✅ Loaded history from experiments/.../history.json
   Total epochs: 35
   Best epoch: 28 (val_loss: 0.0423)
================================================================================
```

---

## Best Practices

### 1. **Use Run All for Everything**

```python
# ✅ GOOD - Works with Run All
config["resume_from_checkpoint"] = True  # Resume existing
SKIP_TRAINING = False  # Will train

# ✅ ALSO GOOD - Also works with Run All
config["resume_from_checkpoint"] = True  # Resume existing
SKIP_TRAINING = True   # Will only load results
```

### 2. **Fresh Training**

```python
config["resume_from_checkpoint"] = False  # Create new experiment
SKIP_TRAINING = False  # Train from scratch
```

### 3. **View Existing Results**

```python
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "latest"
SKIP_TRAINING = True  # Skip training, load results
```

### Problem: "Previous history has invalid structure"

**Cause:** `history.json` is corrupted or missing required keys.

**Solution:** The system automatically starts fresh history tracking. Old data is not used but not deleted.

---

### Problem: "best_model.pth not found"

**Cause:** Training was interrupted before any validation epoch completed.

**Solution:** The system automatically:

1. Searches for emergency checkpoints
2. Searches for periodic checkpoints
3. Copies the most recent one to `best_model.pth`
4. Prints a warning that it may not be the actual "best" model

**Output:**

```
⚠️  best_model.pth not found in .../checkpoints
   📋 Found alternative checkpoint: emergency_checkpoint_oom_epoch_5.pth
   🔄 Copying to best_model.pth...
   ✅ Successfully created best_model.pth
   ⚠️  Note: This may not be the actual best model, just the most recent checkpoint
```

---

### Problem: "Experiment directory mismatch"

**Cause:** Resume loaded a different experiment than expected.

**Output:**

```
⚠️  Warning: Resume experiment mismatch!
   Expected: .../experiments/unet/gaussian/20251230_120000_v1
   Got: .../experiments/unet/gaussian/20251230_150000_v1
   Using: .../experiments/unet/gaussian/20251230_120000_v1
```

**Solution:** This is informational. The system uses the directory set by "Create Output Directories" cell. Verify your `resume_experiment` configuration.

---

### Problem: Inference fails after resume

**Cause:** Previously, inference would look for `best_model.pth` in a newly created experiment.

**Solution:** ✅ **FIXED** - The resume system now:

- Reuses the existing experiment directory
- Ensures `best_model.pth` exists (creates from latest checkpoint if needed)
- Inference finds the model in the correct location

---

## Best Practices

### 1. **Always Use Resume for Interrupted Training**

```python
# ❌ BAD - Creates new experiment, loses history
config["resume_from_checkpoint"] = False

# ✅ GOOD - Continues existing experiment
config["resume_from_checkpoint"] = True
config["resume_experiment"] = "latest"
```

### 3. **Reduce Batch Size After OOM**

```python
# After OOM crash:
config["batch_size"] = 4  # Was 8
config["num_workers"] = 2  # Was 4 (optional, helps on Windows)
config["resume_from_checkpoint"] = True
SKIP_TRAINING = False  # Resume training
```

### 4. **Always Use Run All**

The notebooks are designed to work correctly with "Run All":

- No need to manually select which cells to run
- All variables properly initialized whether training or loading
- `SKIP_TRAINING` controls behavior automatically

```python
config["save_every"] = 5  # Save checkpoint every 5 epochs
```

This ensures recovery options even if `best_model.pth` is lost.

---

## Technical Details

### Files in Experiment Directory

```
experiments/unet/gaussian/20251230_153045_v1/
├── checkpoints/
│   ├── best_model.pth                    # Best validation loss model
│   ├── checkpoint_epoch_005.pth          # Periodic checkpoint
│   ├── checkpoint_epoch_010.pth
│   └── emergency_checkpoint_oom_epoch_7.pth  # OOM crash recovery
├── samples/
│   └── training_samples.png
├── logs/
│   └── tensorboard_logs/
├── config.json                            # Training configuration
└── history.json                           # Metrics history
```

### Checkpoint Contents

```python
checkpoint = {
    "epoch": 15,                          # Completed epoch (1-based)
    "model_state_dict": {...},            # Model weights
    "optimizer_state_dict": {...},        # Optimizer state
    "scheduler_state_dict": {...},        # LR scheduler state
    "metrics": {
        "train": {"loss": 0.045, "l1": 0.032, "ssim": 0.89},
        "val": {"loss": 0.052, "l1": 0.038, "ssim": 0.87}
    }
}
```

### History Structure

```python
history = {
    "train_loss": [0.15, 0.12, 0.10, ...],
    "train_l1": [0.11, 0.09, 0.08, ...],
    "train_ssim": [0.75, 0.80, 0.83, ...],
    "val_loss": [0.14, 0.11, 0.09, ...],
    "val_l1": [0.10, 0.08, 0.07, ...],
    "val_ssim": [0.76, 0.81, 0.85, ...],
    "lr": [0.0001, 0.00009, 0.00008, ...]
}
```

---

## Summary

✅ **What's Fixed:**

1. No more creating new experiments when resuming
2. Automatic `best_model.pth` recovery from emergency checkpoints
3. History consistency across resume sessions
4. Inference works immediately after resume
5. Run All compatible - no optional cells
6. `SKIP_TRAINING` variable for loading results without training

✅ **How to Use:**

1. Set `resume_from_checkpoint=True` (or `False` for fresh)
2. Set `SKIP_TRAINING=True` (to skip) or `False` (to train)
3. Run All - everything works automatically

✅ **When to Resume:**

- OOM crash
- Manual stop (Ctrl+C)
- System crash
- Want to continue training for more epochs
- Want to view history/metrics without training (set `SKIP_TRAINING=True`)

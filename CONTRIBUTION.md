# Thesis Contribution

This repository is a fork of [KeKsBoTer/cinematic-gaussians](https://github.com/KeKsBoTer/cinematic-gaussians) — the official implementation of:

> *Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices*  
> Simon Niedermayr, Christoph Neuhauser, Kaloian Petkov, Klaus Engel, Rüdiger Westermann  
> TU Munich & Siemens Healthineers, 2024 — [arXiv:2404.11285](https://arxiv.org/abs/2404.11285)

All original code belongs to the upstream authors. This fork contains one commit of thesis-specific adaptations on top of that work.

---

## What Was Changed

**Commit:** `b2aecff` — *Adapt cinematic-gaussians for orthographic brain CT visualisation*

### New files added

| File | Purpose |
|---|---|
| `make_cameras.py` | Generates an orthographic camera JSON from a set of PNG brain slice images. Replaces the perspective camera assumptions in the original pipeline with a parallel-projection layout suited to volumetric CT/MRI data. |
| `export_ply.py` | Utility to export a trained `GaussianModel` to PLY format for inspection in external tools (MeshLab, CloudCompare, etc.). |

### Files modified

| File | Change |
|---|---|
| `gaussian_me/io/dataset_readers.py` | Refactored `fetchPly()` to handle explicit xyz/rgb/opacity fields; fixed `readNerfSyntheticInfo()` tuple unpacking and type coercion errors that arose with the brain CT dataset format. |
| `train.py` | Replaced an inline lambda collate function with a named `collate_identity()` function. The lambda caused a Python pickling failure under Windows multiprocessing (`DataLoader` with `num_workers > 0`). |
| `environment.yml` | Replaced the original hand-written minimal spec with a full conda lock file (Python 3.12, PyTorch 2.5.1, CUDA 12.1). Re-encoded from UTF-16 to UTF-8. |
| `.gitignore` | Extended to cover `*.pyd`, egg-info, compiled CUDA extensions, training data folders, and model output folders. |

### Files preserved unchanged (upstream)

The original `(deprecated)environment.yml` (the upstream's hand-written minimal spec) is kept alongside the new full lock file for reference.

---

## Why These Changes Were Needed

The upstream pipeline was designed for perspective-camera scenes (NeRF-synthetic, COLMAP reconstructions). Adapting it for medical CT/MRI data required:

1. **Orthographic cameras** — brain slice stacks have no perspective distortion; `make_cameras.py` generates a camera ring with parallel projection parameters.
2. **Dataset format fixes** — the brain dataset (LADAF-2021-17, 18 µm resolution, JP2 → PNG) exposed edge cases in the PLY loader and NeRF-synthetic reader that did not surface on the original benchmark datasets.
3. **Windows compatibility** — the original codebase was developed on Linux; the lambda pickling issue only appears on Windows due to differences in how `multiprocessing` spawns workers.

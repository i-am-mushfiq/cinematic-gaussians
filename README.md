
<div align="center">

# Cinematic Gaussians — Adapted for Orthographic Brain CT Visualisation

**Thesis adaptation of:**
[Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices](https://arxiv.org/abs/2404.11285)

Simon Niedermayr · Christoph Neuhauser · Kaloian Petkov · Klaus Engel · Rüdiger Westermann  
Technical University of Munich & Siemens Healthineers, 2024

**This fork** — [i-am-mushfiq/cinematic-gaussians](https://github.com/i-am-mushfiq/cinematic-gaussians)  
Upstream — [KeKsBoTer/cinematic-gaussians](https://github.com/KeKsBoTer/cinematic-gaussians)

</div>

---

## Overview

This repository extends the *Cinematic Gaussians* pipeline to support **orthographic volumetric medical datasets** — specifically high-resolution brain CT/MRI stacks acquired in JPEG 2000 (JP2) format. The upstream implementation targets perspective-camera scenes reconstructed via COLMAP or rendered from Blender. This fork adapts the data ingestion, camera representation, and training pipeline to handle:

- Parallel-projection (orthographic) image stacks with known inter-slice spacing
- Medical imaging data in JP2 format (via an external pre-processing pipeline)
- Windows-native execution (fixes multiprocessing pickling failures)
- Full conda environment reproducibility via a pinned lock file

The dataset used is **LADAF-2021-17**, a human brain specimen scanned at two resolutions:
- `Brain1`: 18.048 µm voxel pitch — high-resolution regional volume-of-interest
- `Brain2`: 202.0 µm voxel pitch — whole-organ scan

---

## Thesis Contributions

> All modifications relative to the upstream are contained in a single, reviewable commit.  
> Diff: [KeKsBoTer:master...i-am-mushfiq:master](https://github.com/i-am-mushfiq/cinematic-gaussians/compare/KeKsBoTer:cinematic-gaussians:master...master)

### 1. Orthographic Camera Generator — `make_cameras.py` *(new file)*

The upstream assumes perspective cameras calibrated by COLMAP or supplied as NeRF-synthetic `transform_matrix` JSON. Medical CT/MRI image stacks are acquired with **parallel-projection geometry** — each slice is an orthographic projection of a physical tissue section with known, uniform inter-slice spacing.

`make_cameras.py` generates a `cameras.json` that encodes this geometry directly from a folder of PNG slices:

```python
SLICE_THICK  = 0.02525   # inter-slice spacing in mm (instrument-specific)
ORTHO_FOVY   = 1e-4      # near-zero vertical FOV ≈ orthographic projection
```

Each camera is placed along the Z-axis at `z = idx × slice_thickness`, with identity rotation (looking straight down the stack axis). RGBA images are composited to RGB against a black background before writing. This avoids running COLMAP entirely and produces a geometrically faithful representation of the volumetric acquisition geometry.

**Usage:**
```bash
python make_cameras.py
# Edit IMAGE_DIR, OUTPUT_JSON, SLICE_THICK at the top of the file before running.
# Output: cameras.json consumed directly by train.py via readVolumeSceneInfo()
```

### 2. PLY Exporter — `export_ply.py` *(new file)*

After training, the Gaussian model is serialised as a compressed `.npz` file. The upstream provides no direct path from `.npz` to a universally readable point cloud format. `export_ply.py` loads a trained checkpoint and writes a `.ply` file readable by MeshLab, CloudCompare, or any PLY-compatible viewer:

```bash
python export_ply.py
# Edit npz_path and out_ply at the top of the file.
# Output: a PLY file with per-point xyz + colour + opacity attributes.
```

This is particularly useful for qualitative inspection of Gaussian distributions over anatomical structures prior to compression.

### 3. Dataset Reader Fixes — `gaussian_me/io/dataset_readers.py`

Two issues surfaced when ingesting the brain CT dataset that do not appear on the upstream benchmark scenes:

**`fetchPly()` rewrite:** The original loader assumed all PLY files carry colour as `uint8` RGB. Medical-derived point clouds may carry floating-point or missing fields. The rewrite explicitly extracts `x/y/z`, normalises `red/green/blue` to [0, 1], and initialises opacities to `1.0` when the field is absent — matching the `BasicPointCloud(points, colors, opacities)` signature used downstream.

**`readNerfSyntheticInfo()` tuple unpacking:** `readCamerasFromTransforms()` returns a `(cam_infos, aabb)` tuple. The original call-site unpacked it incorrectly when the return path changed under the thesis modifications, raising a `ValueError` at dataset load time. The fix adds explicit tuple unpacking and type coercion:
```python
train_cam_infos, _ = readCamerasFromTransforms(...)
if isinstance(train_cam_infos, tuple):
    train_cam_infos = list(train_cam_infos)
```

A new scene reader, `readVolumeSceneInfo()`, is also added to handle the `cameras.json` format produced by `make_cameras.py`, routing volume-format datasets through the correct loading path.

### 4. Windows Multiprocessing Fix — `train.py`

The upstream `DataLoader` used an inline lambda as the `collate_fn`. Python's `multiprocessing` module on Windows uses `spawn` (rather than `fork`) to create worker processes, which requires all callable objects passed across process boundaries to be picklable. Lambdas are not picklable under `spawn`, causing an immediate `AttributeError` when `num_workers > 0`.

The fix replaces the lambda with a named module-level function:

```python
def collate_identity(batch):
    return batch
```

This is a silent but critical change for anyone running the pipeline on Windows.

### 5. Full Conda Lock File — `environment.yml`

The upstream `environment.yml` is a minimal hand-written spec (8 packages). This fork replaces it with a **fully pinned conda lock file** generated on the target hardware, covering the complete dependency graph:

- Python 3.12
- PyTorch 2.5.1
- CUDA 12.1 (cudart, nvcc, cuDNN)
- All transitive dependencies with exact build hashes

The original spec is preserved as `(deprecated)environment.yml` for reference. Re-encoding from UTF-16 to UTF-8 also fixes a `conda env create` parse error on non-Windows systems.

---

## Repository Structure

```
cinematic-gaussians/
├── gaussian_me/                  Core Python package
│   ├── model.py                  GaussianModel — xyz, SH coefficients, scale, rotation, opacity
│   ├── renderer.py               CUDA rasterisation via diff-gaussian-rasterization
│   ├── optim.py                  Per-parameter-group learning rate scheduling
│   ├── compression.py            Importance-score vector quantisation
│   ├── eval.py                   PSNR / SSIM metrics
│   ├── args.py                   ModelParams / OptimizationParams / CompressionParams
│   ├── io/
│   │   ├── dataset_readers.py    Scene loaders (COLMAP, NeRF-synthetic, Volume) ← modified
│   │   └── colmap_loader.py      COLMAP binary/text parser
│   └── utils/                    Camera transforms, SH, loss, graphics math
├── train.py                      Training loop                                  ← modified
├── compress.py                   Compression pipeline
├── make_cameras.py               Orthographic camera generator                  ← new
├── export_ply.py                 Checkpoint → PLY exporter                      ← new
├── environment.yml               Full conda lock file                            ← replaced
├── (deprecated)environment.yml   Original upstream minimal spec
├── CONTRIBUTION.md               Detailed change log
└── submodules/
    ├── diff-gaussian-rasterization   CUDA Gaussian rasteriser
    ├── simple-knn                    K-nearest neighbours (CUDA)
    └── simple-nn                     Neural network utilities
```

---

## Installation

Requires an NVIDIA GPU with CUDA 12.1 support and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

```bash
git clone https://github.com/i-am-mushfiq/cinematic-gaussians.git --recursive
cd cinematic-gaussians

conda env create -f environment.yml
conda activate cin3dgs

# Build CUDA extensions
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/simple-nn
```

> **Windows note:** The lock file was generated on Windows 11 with CUDA 12.1. On Linux, use the upstream `(deprecated)environment.yml` as a starting point and resolve the dependency graph for your platform.

---

## Full Pipeline

### Step 1 — Prepare image data

Pre-processed PNG slices are expected in a scene folder with the following layout:

```
<scene_folder>/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── cameras.json        ← generated in Step 2
```

JP2 → PNG conversion is handled by a separate preprocessing pipeline (see the parent repository).

### Step 2 — Generate orthographic cameras

Edit the parameters at the top of `make_cameras.py`:

```python
IMAGE_DIR   = "path/to/scene_folder/images"
OUTPUT_JSON = "path/to/scene_folder/cameras.json"
SLICE_THICK = 0.02525    # mm — check your scanner metadata
ORTHO_FOVY  = 1e-4
```

Then run:

```bash
python make_cameras.py
```

### Step 3 — Train

```bash
python train.py \
    -s <scene_folder> \
    -m <model_output_folder> \
    --eval \
    --test_iterations 7000 15000 30000 \
    --densify_grad_threshold 0.00005 \
    --save_iterations 30000
```

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir <model_output_folder>
```

If VRAM is limited, increase `--densify_grad_threshold` (e.g. `0.0001`) to reduce the number of Gaussians.

### Step 4 — Compress

Reduces the trained model from several hundred MB to typically under 70 MB, while reporting PSNR and SSIM on train/test splits:

```bash
python compress.py \
    -m <model_output_folder> \
    --eval \
    --output_vq <compression_output_folder> \
    --load_iteration 30000
```

### Step 5 — Export to PLY

Inspect the trained Gaussian distribution in MeshLab, CloudCompare, or any PLY viewer:

```bash
# Edit npz_path and out_ply inside export_ply.py, then:
python export_ply.py
```

### Step 6 — Render

```bash
# Single image from a trained or compressed model:
python Rendering_Related_Files/Test_Renderer.py

# Animated preview of all views (matplotlib):
python Rendering_Related_Files/preview.py

# Full PNG sequence:
python Rendering_Related_Files/render_png_sequence.py

# Encode to video:
python Rendering_Related_Files/render_video.py
```

---

## Dataset

**LADAF-2021-17** — Human brain specimen, sourced from the [LADAF project](https://ladaf.univ-lille.fr/).

| Split | Resolution | Format | Description |
|---|---|---|---|
| Brain1 | 18.048 µm/voxel | JP2 | High-res regional VOI (volume-of-interest) |
| Brain2 | 202.0 µm/voxel | JP2 | Whole-organ scan |

Raw JP2 files are not distributed with this repository. Contact the dataset authors for access. The preprocessing pipeline (JP2 → TIF → PNG) is documented in the parent repository.

---

## Results

Training was conducted across multiple experimental runs. Representative outcomes on the brain dataset:

| Model | Iterations | Compressed size | Notes |
|---|---|---|---|
| Test_Model_v2 | 30 000 | ~47 MB | Baseline compression |
| Test_Model_v5 | 30 000 | ~31 MB | Smallest, lower fidelity |
| Test_Model_v7 | 30 000 | ~60 MB | Best quality–size trade-off |

Qualitative output: 281 rendered views at full resolution, encoded as a single MP4 (~40 MB).

---

## Citation

If this work is useful to you, please cite the upstream paper:

```bibtex
@misc{niedermayr2024novel,
    title     = {Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices},
    author    = {Simon Niedermayr and Christoph Neuhauser and Kaloian Petkov and Klaus Engel and Rüdiger Westermann},
    year      = {2024},
    eprint    = {2404.11285},
    archivePrefix = {arXiv},
    primaryClass  = {cs.GR}
}
```

The thesis-specific adaptations in this fork are documented in [CONTRIBUTION.md](CONTRIBUTION.md).

---

## View Selection

The automatic view selection algorithm referenced in the upstream paper is available separately at [chrismile/vpt_denoise — pydens2d](https://github.com/chrismile/vpt_denoise/tree/main/pydens2d).

# PD-MAE-SR Project History & Context

## Project Overview
- **Goal:** Implement Partial Degradation MAE for Super-Resolution.
- **Key Innovation:** Use region-wise realistic degradation instead of random masking to learn restoration-oriented priors.
- **Backbone:** DRCT (Priority 1) / MambaIRv2.

---

## 12/05/2026 — Phase 1: Foundation & Data Pipeline (Completed)
- Established organized directory structure within `PD_MAE_SR/`.
- Implemented complexity masking and visualization.
- Refined data pipeline: Nearest-neighbor upsampling, Gaussian feathering, and multi-level degradation mapping.
- Final Metrics: 75.00% degradation ratio, 0.16s/image performance.

---

## 12/05/2026 — Phase 2: PD-MAE Architecture & Training Refinement (Completed)
- **Architectural Fixes:**
    - Modified `MAEEncoder` to forward **100% of patches** (both clean and degraded), aligning with the PD-MAE novelty.
    - Simplified `MAEDecoder` to process the full encoder output sequence.
    - Implemented and initialized **Sine-Cosine 2D Positional Embeddings** for both encoder and decoder.
- **Training Pipeline Enhancements:**
    - Implemented **Option X Loss**: MSE loss is strictly calculated on the degraded (masked) patches.
    - Integrated **CosineAnnealingLR scheduler** for optimized convergence.
    - Added comprehensive logging (file + stream) and periodic visualization saving.
- **Bug Fixes:**
    - Corrected BGR-to-RGB conversion in `PDMAEDataset`.
    - Fixed visualization saving logic to handle RGB-to-BGR for standard image viewers.
- **Verification:** Ran a **500-iteration smoke test**. 
    - **Final Loss:** 0.015 (Target < 0.030 met).
    - **Visual Confirmation:** Color space is correct, and reconstruction quality is improving rapidly.
- **Result:** Infrastructure is robust, conceptually sound, and **Approved** for full-scale 200k training on ICT Lab Server.


---

## 21/05/2026 — Phase 3: Transition to Stage 2 - HR Structure Fine-Tuning (Completed)
- **HROnlyDataset Implementation:**
    - Created `HROnlyDataset` in `datasets/hr_dataset.py` to train strictly on clean `HR_sub` images.
    - Implemented crop-to-patch logic (`256x256`) and an exact random grid mask (customizable via `--degrade_ratio`, optimized to `50%` instead of `75%`).
    - Blacked out (assigned `0` to BGR channels) the pixels under masked patches to serve as inputs, and set target to fully intact HR patches.
- **Stage 2 Training Customization:**
    - Created `train_stage2.py` in `experiments/train_stage2.py` incorporating new dataset dataloaders.
    - Added `--pretrain` argument to load model weights strictly from Stage 1 Iter 200k checkpoint while resetting optimizer and scheduler states.
    - Adjusted hyperparameters for fine-tuning: `lr = 1e-5` (10x smaller than Stage 1), total iterations `T_max = 100000` (100k), and Cosine Annealing scheduler decaying to `1e-6`.
    - Maintained Option X loss computation strictly targeting blacked-out patches.
    - Added deep weight statistical validation (`verify_loaded_weights`) and a texture-prioritizing visualization selector (`max-variance` selection).
- **Verification & Validation Results:**
    - Confirmed dataset size: **150,877 clean images** in `HR_sub`.
- **Issue Detection & Resolution (Iter 20k with 75% mask):**
    - At iter 20k, visual output showed **complete collapse** — reconstruction was flat gray, no structure.
    - Root cause: 75% random black masking on a full-sequence encoder creates massive **out-of-distribution shift** (encoder never saw black patches in Stage 1).
    - Additionally, visualization was sampling low-texture patches (flat gradients), masking the problem.
    - **Fixes applied:**
        1. Added `verify_loaded_weights()` — deep statistical validation confirming checkpoint load integrity.
        2. Implemented **max-variance visualization selector** — always picks the most textured sample in the batch.
        3. Moved visualization forward pass inside `model.eval()` + `torch.no_grad()` block for accurate output.
        4. Made `--degrade_ratio` configurable via CLI argument.
    - **Decision:** Reduced masking ratio to **50%** (`--degrade_ratio 0.50`).
- **Diagnostic Smoke Test (50% mask, 100 iters):**
    - Weight verification: All 3 key layers **Exact Match: True** vs Stage 1 checkpoint (Iter 200k, Loss 0.0037).
    - **Iter 100 Loss: `0.007503`** (vs `0.016130` at 75% — **2.15x improvement** from mask ratio reduction alone).
    - Visualization shows **actual flower structure reconstructed** with clear petal details and color fidelity — **no gray collapse**.
    - **Result:** Full 100k Stage 2 training launched on ICT Lab Server with 50% masking, lr=1e-5, batch_size=32.

---
## 22/05/2026 — Config Updates for Training

- Added validation dataset (Set14) to `train_pd_mae_realesrgan_x4plus.yml` to enable validation and best‑model checkpointing.
- Added USM flags (`gt_usm`, `l1_gt_usm`, `percep_gt_usm`, `gan_gt_usm`) to the training config to prevent `KeyError` in `realesrgan_model.py`.
- Updated the configuration file accordingly (validation block lines 31‑41, USM flags lines 82‑85).

- Applied the same validation dataset (Set14) and USM flags (`gt_usm`, `l1_gt_usm`, `percep_gt_usm`, `gan_gt_usm`) to `train_pd_mae_stage3.yml`.
- Adjusted learning rate to 5e-5 and total iterations to 200000 for further fine‑tuning.
- Updated the configuration file accordingly (validation block and USM flags lines similar to Stage 2).

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
    - Implemented crop-to-patch logic (`256x256`) and an exact `75%` random grid mask (`768` out of `1024` patches of size `8x8`).
    - Blacked out (assigned `0` to BGR channels) the pixels under masked patches to serve as inputs, and set target to fully intact HR patches.
- **Stage 2 Training Customization:**
    - Created `train_stage2.py` in `experiments/train_stage2.py` incorporating new dataset dataloaders.
    - Added `--pretrain` argument to load model weights strictly from Stage 1 Iter 200k checkpoint while resetting optimizer and scheduler states.
    - Adjusted hyperparameters for fine-tuning: `lr = 1e-5` (10x smaller than Stage 1), total iterations `T_max = 100000` (100k), and Cosine Annealing scheduler decaying to `1e-6`.
    - Maintained Option X loss computation strictly targeting blacked-out patches.
- **Project Tracking & User Guide:**
    - Documented tasks and progress in `task.md` and created `walkthrough.md` mapping out verification commands for the user to run on the server:
      - Dataset image count check: `ls ~/data/dataset/train/HR_sub | wc -l`
      - 100-iteration smoke test command for pipeline validation.
      - Full 100k iteration Stage 2 training command.

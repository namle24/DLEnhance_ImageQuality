# PD-MAE-SR — Project Context & Agent Guide

> **Mục đích file này:** Cung cấp toàn bộ context cho AI agent để triển khai dự án nghiên cứu. Đọc toàn bộ trước khi bắt đầu bất kỳ task nào.

---

## 1. Bối cảnh nhóm nghiên cứu

- **Đơn vị:** University of Science and Technology of Hanoi (USTH), Vietnam Academy of Science and Technology
- **Giảng viên hướng dẫn:** Nguyen Hoang Ha (corresponding author, người đề xuất hướng MAE)
- **Tài nguyên compute:** 1× NVIDIA RTX 3090 (local) + ICT Lab Server (train dài hạn)
- **Framework hiện có:** PyTorch, basicsr, Real-ESRGAN codebase đã setup
- **Paper nền tảng đã có:** "Siamese-based Real-ESRGAN with High-Order Degradation" (KSE 2025)
- **Checkpoints có sẵn:**
  - Stage 1: `~/data/dataset/train/PD_MAE_Checkpoints_Stage1/pd_mae_s1_iter200000.pth`
  - Stage 2: `~/data/dataset/train/PD_MAE_Checkpoints_Stage2/pd_mae_s2_iter100000.pth`
  - **KHÔNG cần train lại Stage 1 và Stage 2**

---

## 2. Paper nền tảng — Siamese Real-ESRGAN (tóm tắt)

### Những gì nhóm đã xây dựng

**Degradation pipeline (điểm mạnh cốt lõi):**
- Two-stage realistic degradation với randomized operation order
- Explicit camera noise: chromatic aberration + sensor noise sigma
- JPEG compression với narrower ranges
- Pre-generated LQ-HQ pairs offline
- **Multi-level degradation: 4 mức 90% / 80% / 70% / 60%** từ mỗi GT image

**Siamese architecture:**
- Weight-sharing dual-branch (Teacher + Student dùng chung params RRDB)
- Teacher xử lý LQ mild (ít degraded), Student xử lý LQ severe (nhiều degraded)
- Teacher branch: `torch.no_grad()`, output dùng làm pseudo-target
- Three-phase curriculum: Phase1=90/80, Phase2=80/70, Phase3=70/60
- ~250k iterations/phase, tổng 800k iterations

**Loss function:**
```
L_total = w_pixel * L_pixel        (L1, weight=2.0)
        + w_perceptual * L_perceptual  (VGG19, Johnson et al. 2016)
        + w_GAN * L_GAN              (vanilla GAN, weight=0.5)
        + w_out_distill * L_out_distill   (λ=0.15)
        + w_feat_distill * L_feat_distill  (λ=0.15)
        + consistency regularization     (weight=0.075)
```

**Backbone:** RRDBNet (23 RRDB blocks, 64 feature channels) + UNet discriminator with spectral normalization

**Dataset:** DF2K (DIV2K + Flickr2K + OutdoorScene) + Unsplash + RealSR

**Kết quả tốt nhất (Phase 2):**
- RealSR V3 Canon NIQE: 5.71 (baseline Real-ESRGAN: 6.88 → cải thiện ~17%)
- Urban100 NIQE: 3.82
- PSNR drop so với baseline được justify bằng perception-distortion tradeoff [Blau & Michaeli 2018]

---

## 3. Ý tưởng mới — PD-MAE-SR

### 3.1 Core insight (từ thầy Nguyen Hoang Ha)

> MAE phát huy tác dụng khi **không có GT**. Khi đã có GT rồi thì MAE không cần thiết theo nghĩa gốc. Tuy nhiên, nhóm có thể dùng MAE theo cách **khác với mục đích ban đầu**: không phải để tận dụng unlabeled data, mà để học **loại representation đặc biệt mà supervised training không học được**.

**Partial Degradation** là core idea:
- MAE gốc: mask ngẫu nhiên patch → reconstruct pixels → học representation
- Ý tưởng thầy: **degrade một phần ảnh** (75%) thay vì mask hoàn toàn → reconstruct HR → học restoration prior

| | MAE gốc | PD-MAE |
|--|---------|--------|
| "Mask" là gì | Xóa patch, thay bằng token rỗng | Degrade patch bằng noise/blur/JPEG |
| Target reconstruct | Pixel gốc của patch | HR patch sạch (GT) |
| Học được gì | Image representation | Restoration prior |
| Cần GT lúc pretrain | Không | Có (nhưng nhóm đã có sẵn) |

### 3.2 Tên framework: PD-MAE-SR
**(Partial Degradation MAE for Super-Resolution)**

### 3.3 Architecture tổng thể

```
┌──────────────────────────────────────────────────────────┐
│  STAGE 1 PRETRAIN: Partial Degradation MAE               │
│                                                          │
│  HR patch                                                │
│    → Region selection (75% vùng theo gradient map)       │
│    → Apply degradation pipeline lên 75% vùng đó         │
│    → 25% còn lại giữ nguyên HR                           │
│                                                          │
│  MAE Encoder + Decoder                                   │
│  Input: partially degraded image                         │
│  Target: HR GT                                           │
│  → Encoder học: restoration-oriented prior               │
└────────────────────────┬─────────────────────────────────┘
                         ↓ encoder checkpoint
┌──────────────────────────────────────────────────────────┐
│  STAGE 2 FINE-TUNE: HR Structure MAE                     │
│                                                          │
│  Input: HR masked randomly 75% (clean, no degradation)   │
│  Target: HR GT                                           │
│  Learning rate: ÷10 so với stage 1                       │
│  → Encoder re-calibrate về HR texture domain             │
└────────────────────────┬─────────────────────────────────┘
                         ↓ encoder frozen
┌──────────────────────────────────────────────────────────┐
│  STAGE 3 DOWNSTREAM: PD-MAE encoder + SR backbone        │
│                                                          │
│  LQ → PD-MAE Encoder (frozen) → structural prior f_mae  │
│                ↓ SFT injection vào SR backbone           │
│  LQ → SR Backbone (DRCT hoặc MambaIRv2) → SR output     │
│                                                          │
│  L_total = L_pixel + L_perceptual                        │
│          + L_GAN (nếu dùng GAN backbone)                 │
│          + L_PD_MAE (MAE encoder consistency loss)       │
└──────────────────────────────────────────────────────────┘
```

### 3.4 Region selection strategy

**Bắt đầu với Gradient-based (đơn giản, không cần train thêm):**
```python
def compute_complexity_mask(hr_patch, degrade_ratio=0.75):
    gray = rgb_to_gray(hr_patch)
    grad_x = sobel(gray, axis=0)
    grad_y = sobel(gray, axis=1)
    complexity = normalize(sqrt(grad_x² + grad_y²))
    # Mask top degrade_ratio% complexity patches
    threshold = percentile(complexity, (1 - degrade_ratio) * 100)
    mask = (complexity > threshold)
    lq_partial = apply_degradation(hr_patch, mask,
                                   level=random.choice([60,70,80,90]))
    return lq_partial, hr_patch
```

**Sau đó ablate với SLIC superpixel** (semantic regions hơn, không cần train):
```python
from skimage.segmentation import slic
segments = slic(hr_patch, n_segments=50, compactness=10)
# Degrade các segment có high gradient complexity
```

**Không làm learned RNN segmentation** vì không có train signal rõ ràng → risk cao.

### 3.5 Backbone candidate (theo thứ tự ưu tiên)

1. **DRCT** (~14M params) — SOTA tại NTIRE 2024/2025, code public, train được trên RTX 3090. **Ưu tiên số 1. ĐÂY LÀ BACKBONE CHÍNH CHO STAGE 3.**
2. **MambaIRv2** (~12M params) — CVPR 2025, efficient hơn nhưng cần check memory trước. Backup nếu DRCT không đủ tốt.

> ⚠️ **KHÔNG dùng Real-ESRGAN (RRDBNet) làm SR backbone nữa.** Lý do chi tiết trong Section 12.

### 3.6 Novelty statement (1 câu cho reviewer)

> *"We propose Partial Degradation MAE (PD-MAE), a self-supervised pretraining strategy that replaces random masking with region-wise realistic degradation, enabling the encoder to learn restoration-oriented priors — and integrate this encoder as structural guidance into a SOTA SR backbone via Spatial Feature Transform injection."*

---

## 4. Ablation study cần thiết

Đây là bảng ablation mục tiêu — **mỗi row phải có số liệu thực**:

| Config | Region mask | MAE stage | Backbone | NIQE RealSR V3↓ | PSNR Set5↑ | SSIM↑ |
|--------|-------------|-----------|----------|-----------------|-----------|-------|
| Real-ESRGAN official | — | — | RRDBNet | 4.51 ✅ | — | — |
| DRCT baseline | — | — | DRCT | ? | ? | ? |
| + Random MAE | Random | Stage 1 only | DRCT | ? | ? | ? |
| + Gradient PD-MAE | Gradient | Stage 1 only | DRCT | ? | ? | ? |
| + 2-stage PD-MAE | Gradient | Stage 1 + 2 | DRCT | ? | ? | ? |
| **PD-MAE-DRCT (full)** | Gradient | Stage 1 + 2 + SFT | DRCT | ? | ? | ? |
| PD-MAE-DRCT + SLIC | SLIC | Stage 1 + 2 + SFT | DRCT | ? | ? | ? |

> **Quan trọng:** Row "DRCT baseline" phải được chạy trước và song song với PD-MAE-DRCT để có fair comparison. Không thể claim improvement nếu không có DRCT baseline số liệu thực.

---

## 5. Roadmap triển khai (cập nhật 25/05/2026)

### ✅ ĐÃ HOÀN THÀNH

**Stage 1 — PD-MAE Pretraining (DONE)**
- 200k iterations, final loss 0.0037
- Encoder học được restoration-oriented representation tốt
- Checkpoint: `pd_mae_s1_iter200000.pth`

**Stage 2 — HR Fine-tuning (DONE)**
- 100k iterations, final loss 0.0068
- Encoder re-calibrated về HR texture domain
- Checkpoint: `pd_mae_s2_iter100000.pth`

**Stage 3A — Real-ESRGAN backbone (FAILED, ABANDONED)**
- Kết quả: NIQE 8.08 vs baseline 4.51 — tệ hơn baseline 3.57 points
- Nguyên nhân: data mismatch + catastrophic forgetting
- Chi tiết: xem `PD_MAE_SR_Stage3_Issue_Report.md`

### 🔄 ĐANG LÀM — Stage 3B: DRCT backbone

**Bước 1 — Setup DRCT:**
```bash
git clone https://github.com/ming053l/DRCT
cd DRCT
pip install -r requirements.txt
# Đọc kỹ DRCT architecture trước khi implement bất cứ gì
```

**Bước 2 — Hiểu DRCT architecture:**
- Xác định số Dense-Residual Connected Transformer Groups (DCTG)
- Xác định channel dimension tại mỗi stage
- Xác định injection points phù hợp (không phải RRDB blocks nữa)
- **Báo cáo reviewer trước khi implement SFT**

**Bước 3 — Implement SFT cho DRCT:**
- SFT layer giữ nguyên design (scale + shift)
- Channel projection: 384 → DRCT_channels (cần xác định sau bước 2)
- Inject sau mỗi DCTG group (tương tự mỗi 5 RRDB blocks trước)

**Bước 4 — Training strategy (THAY ĐỔI QUAN TRỌNG so với Stage 3A):**
- Dùng **DRCT's own training pipeline và degradation** — không dùng LR_all_sub
- PD-MAE encoder frozen, inject features qua SFT
- L_PD_MAE weight: bắt đầu với 0.05 (thấp hơn 0.1 trước đó)
- Train song song DRCT baseline (không MAE) để có fair comparison

**Bước 5 — Evaluation:**
- Benchmarks: Set5, Set14, RealSR V3 Canon, RealSR V3 Nikon, BSDS100, Urban100
- Primary metric: NIQE (lower = better)
- Secondary: PSNR, SSIM
- Compare với: DRCT baseline, Real-ESRGAN official (4.51), paper Siamese results

### ⏳ CHỜ — Paper Writing
- Target venue: IEEE TIP hoặc IEEE TCSVT
- Sections cần viết: Abstract, Intro, Related Work, Methodology, Experiments, Conclusion

---

## 6. Related work cần cite (đã verified)

### Core MAE papers
- He et al. (CVPR 2022) — MAE gốc **[bắt buộc]**
- Bengio et al. (ICML 2009) — Curriculum learning **[bắt buộc cho 2-stage]**

### SR backbone papers
- Wang et al. (ICCV 2021) — Real-ESRGAN **[baseline]**
- Liang et al. (ICCV 2021) — SwinIR **[comparison]**
- Hsu et al. (CVPR 2024) — DRCT **[proposed backbone]**
- Guo et al. (ECCV 2025) — MambaIRv2 **[alternative backbone]**

### Degradation & blind SR
- Zhang et al. (ICCV 2021) — BSRGAN **[comparison]**
- Blau & Michaeli (CVPR 2018) — Perception-distortion tradeoff **[justify PSNR drop]**

### MAE + Restoration (gap analysis)
- Qin et al. (ECCV 2024) — RAM: Restore Anything with Masks **[most related, làm all-in-one restoration không phải SR]**
- Zhou et al. (2023) — MAE as perceptual loss **[related approach]**

### Segmentation cho masking
- Achanta et al. (TPAMI 2012) — SLIC superpixel **[nếu dùng SLIC ablation]**

---

## 7. Các quyết định đã được xác nhận

| Quyết định | Lý do | Ngày |
|-----------|-------|------|
| Không dùng diffusion làm backbone | Mâu thuẫn với mục tiêu efficient/lightweight | 12/05 |
| Không dùng Transformer nặng (HAT-L) | RTX 3090 không đủ | 12/05 |
| Không làm learned segmentation | Không có train signal rõ; risk cao | 12/05 |
| Dùng gradient map cho masking | Đơn giản, argument rõ, không cần dependency thêm | 12/05 |
| 2-stage MAE (LQ→HR trước, HR fine-tune sau) | Curriculum learning: học restoration trước, refine sau | 12/05 |
| Encoder frozen khi train SR | Không tăng inference cost | 12/05 |
| **KHÔNG dùng Real-ESRGAN làm SR backbone** | Data mismatch, catastrophic forgetting, NIQE 8.08 vs baseline 4.51 | **25/05** |
| **Dùng DRCT làm SR backbone** | SOTA 2024, training stable, không GAN, generalizable | **25/05** |
| **Dùng DRCT's own training pipeline** | Tránh data mismatch như Stage 3A | **25/05** |
| **L_PD_MAE weight: 0.05** | 0.1 trước đó không cause harm nhưng giảm conservative hơn | **25/05** |

---

## 8. Các rủi ro cần theo dõi

| Rủi ro | Mức độ | Cách xử lý |
|--------|--------|-----------|
| ViT-Small encoder quá nặng để inject vào DRCT | Trung bình | Thêm linear projection 384→DRCT_channels trước SFT |
| L_PD_MAE conflict với DRCT loss | Thấp | Bắt đầu với weight=0.05, monitor loss curve |
| DRCT injection points không rõ ràng | Trung bình | Đọc kỹ code trước, báo cáo reviewer trước khi implement |
| Kết quả không beat DRCT baseline | Trung bình | Thử cross-attention thay SFT; adjust injection points |
| **[ĐÃ XẢY RA] Data mismatch với Real-ESRGAN** | — | Giải quyết: dùng backbone's own training pipeline | 

---

## 9. Vai trò các bên trong quá trình triển khai

**AI Agent (coding/implementation):**
- Implement các module theo spec trong file này
- Chạy experiments theo thứ tự trong roadmap
- Report kết quả sau mỗi checkpoint

**Claude (reviewer/advisor — người tạo file này):**
- Đánh giá kết quả mỗi checkpoint
- Cảnh báo khi đi lệch hướng
- Điều chỉnh strategy nếu kết quả không như kỳ vọng
- Review code trước khi chạy experiment tốn compute
- Đảm bảo paper argument nhất quán với kết quả

**Nhóm nghiên cứu:**
- Quyết định cuối cùng về hướng đi
- Chạy experiments thực tế trên RTX 3090
- Viết paper

---

## 10. Quy tắc làm việc với agent

1. **Trước khi implement bất kỳ module mới:** Confirm spec với reviewer trước, không tự ý thay đổi architecture
2. **Sau mỗi checkpoint:** Report kết quả số liệu cụ thể (không chỉ "có vẻ tốt")
3. **Nếu kết quả tệ hơn baseline:** Không panic, report và chờ hướng dẫn điều chỉnh
4. **Ablation phải chạy đủ tất cả variants** trong bảng Section 4 — không skip row nào
5. **Visualization là bắt buộc** tại mỗi checkpoint — không chỉ có số liệu

---

## 11. Câu hỏi mở chưa có câu trả lời (cần experiment để quyết định)

- Injection points trong DRCT: sau mỗi DCTG group hay chỉ một số?
- Channel dimension SFT: 384 → DRCT_channels (cần đọc code DRCT)
- L_PD_MAE weight tối ưu: 0.05 hay cần tune thêm?
- Dùng single-layer hay multi-layer encoder output cho SFT?
- DRCT baseline cần bao nhiêu iters để converge trên data của nhóm?

---

## 12. ⚠️ Bài học từ Stage 3A — KHÔNG LẶP LẠI

> **Agent phải đọc section này trước khi bắt đầu bất kỳ implementation nào cho Stage 3B.**

### Những gì đã xảy ra

Stage 3A dùng Real-ESRGAN (RRDBNet) làm backbone, train trên LR_all_sub (4 degradation levels cố định). Sau 130k iterations:

- NIQE trên RealSR V3 Canon: **8.08** (baseline official: 4.51)
- Model làm ảnh **xấu hơn** cả LR input (LR NIQE: 7.37)
- Ablation tắt L_PD_MAE: NIQE chỉ giảm từ 8.08 → 7.96 (không phải root cause)

### Root cause

**Data mismatch:** LR_all_sub chỉ có 4 degradation patterns cố định. Real-ESRGAN official train với online degradation (vô số variants). RealSR V3 là real-world data → model overfit vào 4 patterns → generalize kém → artifacts → NIQE cao.

**Catastrophic forgetting:** Pretrained weights từ RealESRNet_x4plus (train trên DF2K online degradation) bị overwrite bởi 130k iters trên distribution hẹp.

### Quy tắc bắt buộc cho Stage 3B

```
QUY TẮC 1: LUÔN dùng backbone's own training pipeline và degradation.
           KHÔNG ép backbone train trên LR_all_sub.

QUY TẮC 2: PD-MAE encoder là FROZEN FEATURE EXTRACTOR.
           Nó inject prior, không thay thế training data.

QUY TẮC 3: Phải chạy DRCT baseline (không MAE) TRƯỚC hoặc SONG SONG.
           Không thể claim improvement nếu không có baseline số liệu thực.

QUY TẮC 4: Evaluate trên RealSR V3 Canon sau mỗi 50k iters.
           Nếu NIQE > 6.0 tại iter 50k → dừng ngay, báo reviewer.
```

---

*File này được tạo ngày 12/05/2026, cập nhật lần cuối 25/05/2026 sau khi pivot từ Real-ESRGAN sang DRCT backbone.*

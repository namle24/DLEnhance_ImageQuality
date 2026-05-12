# PD-MAE-SR — Project Context & Agent Guide

> **Mục đích file này:** Cung cấp toàn bộ context cho AI agent để triển khai dự án nghiên cứu. Đọc toàn bộ trước khi bắt đầu bất kỳ task nào.

---

## 1. Bối cảnh nhóm nghiên cứu

- **Đơn vị:** University of Science and Technology of Hanoi (USTH), Vietnam Academy of Science and Technology
- **Giảng viên hướng dẫn:** Nguyen Hoang Ha (corresponding author, người đề xuất hướng MAE)
- **Tài nguyên compute:** 1× NVIDIA RTX 3090 (24GB VRAM)
- **Framework hiện có:** PyTorch, training pipeline Real-ESRGAN đã setup sẵn
- **Paper nền tảng đã có:** "Siamese-based Real-ESRGAN with High-Order Degradation" (đã submit/publish tại KSE 2025)

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

1. **DRCT** (~14M params) — SOTA tại NTIRE 2024/2025, code public, train được trên 1 RTX 3090. **Ưu tiên số 1.**
2. **MambaIRv2** (~12M params) — CVPR 2025, efficient hơn nhưng cần check memory trước.
3. **StarSRGAN** (~16M params) — nhóm đã quen, fallback nếu DRCT khó tích hợp.

### 3.6 Novelty statement (1 câu cho reviewer)

> *"We propose Partial Degradation MAE (PD-MAE), a self-supervised pretraining strategy that replaces random masking with region-wise realistic degradation, enabling the encoder to learn restoration-oriented priors — and integrate this encoder as structural guidance into a SOTA SR backbone via Spatial Feature Transform injection."*

---

## 4. Ablation study cần thiết

Đây là bảng ablation mục tiêu — **mỗi row phải có số liệu thực**:

| Config | Region mask | MAE stage | Backbone | NIQE↓ | PSNR↑ | SSIM↑ |
|--------|-------------|-----------|----------|-------|-------|-------|
| DRCT baseline | — | — | DRCT | ? | ? | ? |
| + Random MAE (MAE gốc) | Random | Stage 1 only | DRCT | ? | ? | ? |
| + Gradient PD-MAE | Gradient | Stage 1 only | DRCT | ? | ? | ? |
| + 2-stage PD-MAE | Gradient | Stage 1 + 2 | DRCT | ? | ? | ? |
| **PD-MAE-SR (full)** | Gradient | Stage 1 + 2 + SFT | DRCT | ? | ? | ? |
| PD-MAE-SR + SLIC | SLIC | Stage 1 + 2 + SFT | DRCT | ? | ? | ? |

---

## 5. Roadmap triển khai

### Tuần 1-2 — Foundation
- [ ] Implement `compute_complexity_mask()` với Sobel gradient
- [ ] Visualize region map trên 50 ảnh DF2K — confirm phân vùng hợp lý
- [ ] Compare visual: random mask vs gradient mask vs SLIC
- [ ] **Checkpoint:** Figure 1 của paper (visualization of partial degradation)

### Tuần 3-4 — Stage 1 MAE Pretrain
- [ ] Setup MAE với ViT-Small backbone (patch_size=8, encoder_dim=384, decoder_dim=192)
- [ ] Dataset: tất cả LQ-HR pairs từ degradation pipeline (multi-level 60/70/80/90%)
- [ ] Train ~200k iterations với lr=1e-4, AdamW
- [ ] **Checkpoint:** Reconstruction quality trên validation set

### Tuần 4-5 — Stage 2 Fine-tune
- [ ] Fine-tune encoder trên HR-only data (DF2K HR), random mask 75%
- [ ] lr=1e-5 (÷10), ~100k iterations
- [ ] Visualize t-SNE của encoder features: stage 1 vs stage 2
- [ ] **Checkpoint:** Figure 2 của paper (feature space visualization)

### Tuần 5-7 — Integration với SR backbone
- [ ] Implement SFT injection layer
- [ ] Integrate PD-MAE encoder (frozen) vào DRCT
- [ ] Thêm L_PD_MAE consistency loss
- [ ] Train full model, so sánh với DRCT baseline
- [ ] **Checkpoint:** Preliminary results trên RealSR V3

### Tuần 7-8 — Ablation & Evaluation
- [ ] Chạy đủ 6 variants trong ablation table
- [ ] Evaluate trên Set5, Set14, RealSR V3 (Canon + Nikon), Urban100, BSDS100
- [ ] Qualitative comparison: visualize SR outputs
- [ ] **Checkpoint:** Full results table

### Tuần 9-10 — Paper Writing
- [ ] Draft paper theo IEEE double-column format
- [ ] Sections: Abstract, Intro, Related Work, Methodology, Experiments, Conclusion
- [ ] Target venue: IEEE TIP hoặc TCSVT (journal); ICCV workshop nếu nhanh

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

| Quyết định | Lý do |
|-----------|-------|
| Không dùng diffusion làm backbone | Mâu thuẫn với mục tiêu efficient/lightweight; inference nặng |
| Không dùng Transformer nặng (HAT-L) | RTX 3090 không đủ; không phải mục tiêu nhóm |
| Không làm learned segmentation | Không có train signal rõ; risk cao; không cần thiết |
| Dùng gradient map trước | Đơn giản, argument rõ, không cần thêm dependency |
| 2-stage MAE (LQ→HR trước, HR fine-tune sau) | Curriculum: học restoration trước, rồi refine về HR domain |
| Encoder frozen khi train SR | Không tăng inference cost; MAE chỉ dùng lúc train |

---

## 8. Các rủi ro cần theo dõi

| Rủi ro | Mức độ | Cách xử lý |
|--------|--------|-----------|
| ViT-Small encoder quá nặng để inject vào DRCT | Trung bình | Thử channel reduction layer trước SFT |
| Stage 2 fine-tune làm encoder "quên" degradation prior | Trung bình | Monitor reconstruction quality trên LQ data sau stage 2 |
| L_PD_MAE conflict với L_perceptual | Thấp | Tune weights, có thể disable 1 trong 2 |
| DRCT không support SFT injection dễ dàng | Thấp | Fallback về StarSRGAN nếu cần |
| Kết quả không beat DRCT baseline | Trung bình | Adjust injection strategy, thử cross-attention thay SFT |

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

- Tỉ lệ degrade tối ưu: 75% hay cần tune?
- Inject MAE features vào tất cả RRDB blocks hay chỉ một số? (Thử: mỗi 5 blocks)
- Channel dimension cho SFT layer? (Thử: linear projection từ 384 → 64)
- Mask ratio tối ưu cho Stage 2: 75% (như MAE gốc) hay thấp hơn cho SR?
- Có nên dùng MAE features từ multiple layers hay chỉ final encoder output?

---

*File này được tạo ngày 12/05/2026 từ cuộc thảo luận nghiên cứu giữa nhóm USTH và Claude (Anthropic). Cập nhật file này sau mỗi quyết định quan trọng.*

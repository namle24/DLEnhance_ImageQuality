# PD-MAE-SR — Phác thảo ý tưởng & Kế hoạch

> Đọc file này trước mỗi buổi làm việc. Cập nhật phần **Trạng thái hiện tại** sau mỗi session.

---

## Ý tưởng cốt lõi — Một câu

> Thay vì mask ngẫu nhiên như MAE gốc, ta **degrade một phần ảnh** (75%) bằng degradation pipeline thực tế, buộc encoder học cách "nhìn qua" noise/blur/JPEG để hiểu HR structure — rồi dùng encoder đó làm prior guide cho SR backbone SOTA.

---

## Tại sao ý tưởng này có giá trị

| Vấn đề hiện tại | Giải pháp của PD-MAE |
|----------------|---------------------|
| MAE gốc học representation chung, không biết gì về degradation | PD-MAE học restoration prior — encoder hiểu noise/blur là gì |
| VGG perceptual loss học từ ImageNet (classification), không liên quan SR | PD-MAE encoder học trực tiếp từ LQ→HR pairs |
| DRCT/MambaIR mạnh nhưng không có degradation-aware guidance | PD-MAE encoder inject structural prior vào backbone qua SFT |
| MAE cần unlabeled data → nhóm đã có GT sẵn | Dùng GT để học restoration prior tốt hơn, không lãng phí |

---

## Framework — 3 stages

```
STAGE 1 ── Partial Degradation MAE Pretrain
│
│  HR patch → chọn 75% vùng phức tạp (Sobel gradient map)
│           → degrade 75% vùng đó (pipeline: noise/blur/JPEG/camera)
│           → giữ nguyên 25% còn lại
│
│  MAE Encoder + Decoder
│  Input : partially degraded image
│  Target: HR GT
│  Học   : "nhìn LQ bị hỏng một phần → suy ra HR đầy đủ"
│
▼
STAGE 2 ── HR Fine-tune MAE
│
│  Input : HR clean bị mask ngẫu nhiên 75%
│  Target: HR GT
│  LR    : ÷10 so với Stage 1
│  Học   : re-calibrate encoder về HR texture domain
│
▼
STAGE 3 ── PD-MAE encoder + SR Backbone (DRCT)
│
│  LQ → PD-MAE Encoder [FROZEN] → features f_mae
│                ↓ SFT injection
│  LQ → DRCT backbone → SR output
│
│  Loss = L_pixel + L_perceptual + L_GAN + L_PD_MAE
```

---

## So sánh với các paper liên quan — Chỗ khác biệt

| Paper | Làm gì | Khác với PD-MAE |
|-------|--------|----------------|
| MAE (He 2022) | Random mask → reconstruct pixels | PD-MAE degrade thay vì mask, target là HR |
| RAM (ECCV 2024) | Mask → all-in-one restoration | Không phải SR, không có partial degradation |
| SP-IGAN (2025) | Semantic prior từ segmentation model | Cần labeled data, không self-supervised |
| DRCT (CVPR 2024) | SR backbone mạnh | Không có degradation-aware encoder prior |
| **PD-MAE-SR** | **Tất cả ở trên kết hợp** | **Chưa ai làm** |

---

## Backbone được chọn: DRCT

- ~14M params, train được trên 1× RTX 3090
- SOTA tại NTIRE 2024/2025
- Code public: `https://github.com/ming053l/DRCT`
- Fallback: StarSRGAN (nhóm đã quen), MambaIRv2 (nếu memory ổn)

---

## Degradation pipeline (tài sản của nhóm)

Pipeline **đã có sẵn** từ paper Siamese — đây là lợi thế so với mọi paper khác:
- Randomized degradation order
- Camera noise: chromatic aberration + sensor noise sigma  
- JPEG narrower ranges
- **Multi-level: 90% / 80% / 70% / 60%** từ mỗi GT image
- Pre-generated offline (reproducible)

---

## Ablation table mục tiêu

| # | Config | Mask | MAE Stage | NIQE↓ | PSNR↑ |
|---|--------|------|-----------|-------|-------|
| 1 | DRCT baseline | — | — | | |
| 2 | + Random MAE | Random | Stage 1 | | |
| 3 | + Gradient PD-MAE | Gradient | Stage 1 | | |
| 4 | + 2-stage PD-MAE | Gradient | Stage 1+2 | | |
| 5 | **PD-MAE-SR (full)** | Gradient | Stage 1+2+SFT | | |
| 6 | + SLIC variant | SLIC | Stage 1+2+SFT | | |

---

## Roadmap — 10 tuần

```
Tuần 1-2   Gradient mask → visualize → Figure 1 paper
Tuần 3-4   Stage 1 MAE pretrain (200k iter, ViT-Small)
Tuần 4-5   Stage 2 fine-tune (100k iter) → t-SNE viz → Figure 2
Tuần 5-7   Integrate encoder vào DRCT qua SFT → train full model
Tuần 7-8   Ablation (6 variants) + evaluation (6 benchmarks)
Tuần 9-10  Viết paper → target IEEE TIP / TCSVT
```

### Benchmarks cần evaluate
Set5 · Set14 · RealSR V3 Canon · RealSR V3 Nikon · BSDS100 · Urban100

---

## Câu hỏi mở — Trả lời bằng experiment

- [ ] Tỉ lệ degrade tối ưu: 75% hay cần tune?
- [ ] Inject vào mỗi 5 RRDB blocks hay tất cả?
- [ ] SFT channel: 384 → 64 có đủ không?
- [ ] Mask ratio Stage 2: 75% hay thấp hơn?
- [ ] Dùng single-layer hay multi-layer encoder output?

---

## Rủi ro & cách xử lý

| Rủi ro | Xử lý |
|--------|-------|
| ViT-Small encoder quá nặng khi inject | Thêm linear projection 384→64 trước SFT |
| Stage 2 làm encoder "quên" degradation | Monitor recon quality trên LQ sau stage 2 |
| DRCT khó tích hợp SFT | Fallback StarSRGAN |
| Kết quả không beat DRCT | Thử cross-attention thay SFT, điều chỉnh injection point |

---

## Key references

```
[1] He et al. - MAE (CVPR 2022)
[2] Wang et al. - Real-ESRGAN (ICCV 2021)
[3] Hsu et al. - DRCT (CVPR 2024)
[4] Guo et al. - MambaIRv2 (CVPR 2025)
[5] Liang et al. - SwinIR (ICCV 2021)
[6] Zhang et al. - BSRGAN (ICCV 2021)
[7] Qin et al. - RAM (ECCV 2024)         ← most related, khác ở SR + partial degrad
[8] Bengio et al. - Curriculum Learning (ICML 2009)
[9] Blau & Michaeli - Perception-Distortion Tradeoff (CVPR 2018)
[10] Johnson et al. - Perceptual Loss (ECCV 2016)
[11] Nhóm - Siamese Real-ESRGAN (KSE 2025) ← paper nền tảng
```

---

## Novelty statement (dùng cho Abstract & Introduction)

> *"We propose Partial Degradation MAE (PD-MAE), a self-supervised pretraining strategy that replaces random masking with region-wise realistic degradation to learn restoration-oriented structural priors. Unlike prior MAE approaches that reconstruct clean pixels from masked inputs, PD-MAE bridges the sim-to-real gap by training the encoder to recover HR content from partially degraded observations — directly aligned with the blind SR objective. The pretrained encoder is then integrated as frozen structural guidance into a SOTA SR backbone via Spatial Feature Transform injection, achieving superior perceptual quality without additional inference cost."*

---

## Trạng thái hiện tại

```
[ ] Tuần 1  — Chưa bắt đầu
[ ] Tuần 2  — Chưa bắt đầu
[ ] Tuần 3  — Chưa bắt đầu
...

Lần cập nhật cuối: 12/05/2026
Người cập nhật: ___
Ghi chú: ___
```

---

*Khi cần tư vấn hoặc review, paste file này + kết quả mới nhất vào Claude.*  
*File context đầy đủ hơn: `PD_MAE_SR_project_context.md`*

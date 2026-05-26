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
STAGE 3 ── PD-MAE encoder + SR Backbone (**DRCT** — KHÔNG phải Real-ESRGAN)
│
│  LQ → PD-MAE Encoder [FROZEN] → features f_mae
│                ↓ SFT injection vào DRCT groups
│  LQ → DRCT backbone → SR output
│
│  Loss = DRCT's own losses + 0.05 × L_PD_MAE
│  Training data = DRCT's own pipeline (KHÔNG dùng LR_all_sub)
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

## Backbone được chọn: DRCT ✅ (confirmed 25/05/2026)

- ~14M params, train được trên RTX 3090
- SOTA tại NTIRE 2024/2025, training stable (không GAN)
- Code public: `https://github.com/ming053l/DRCT`
- Fallback: MambaIRv2 nếu DRCT không đủ tốt

> ⚠️ **Real-ESRGAN đã bị loại bỏ** — Stage 3A với RRDBNet cho NIQE 8.08 vs baseline 4.51. Root cause: data mismatch + catastrophic forgetting. Chi tiết: `PD_MAE_SR_Stage3_Issue_Report.md`

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

| # | Config | Mask | MAE Stage | NIQE RealSR V3↓ | PSNR↑ |
|---|--------|------|-----------|-----------------|-------|
| 0 | Real-ESRGAN official | — | — | **4.51** ✅ | — |
| 1 | DRCT baseline | — | — | ? | ? |
| 2 | + Random MAE | Random | Stage 1 | ? | ? |
| 3 | + Gradient PD-MAE | Gradient | Stage 1 | ? | ? |
| 4 | + 2-stage PD-MAE | Gradient | Stage 1+2 | ? | ? |
| 5 | **PD-MAE-DRCT (full)** | Gradient | Stage 1+2+SFT | ? | ? |
| 6 | + SLIC variant | SLIC | Stage 1+2+SFT | ? | ? |

---

## Roadmap (cập nhật 25/05/2026)

```
✅ DONE  Stage 1 MAE pretrain     200k iters, loss=0.0037
✅ DONE  Stage 2 HR fine-tune     100k iters, loss=0.0068
❌ FAIL  Stage 3A Real-ESRGAN     NIQE 8.08, abandoned
🔄 NOW   Stage 3B DRCT backbone   Setup → Implement SFT → Train → Eval
⏳ NEXT  Ablation (6 variants)    Sau khi có PD-MAE-DRCT kết quả
⏳ NEXT  Paper writing            IEEE TIP / TCSVT
```

### Checkpoints có sẵn — KHÔNG cần train lại
```
Stage 1: ~/data/dataset/train/PD_MAE_Checkpoints_Stage1/pd_mae_s1_iter200000.pth
Stage 2: ~/data/dataset/train/PD_MAE_Checkpoints_Stage2/pd_mae_s2_iter100000.pth
```

### Bước tiếp theo ngay bây giờ
```
1. git clone https://github.com/ming053l/DRCT
2. Đọc DRCT architecture code — report channel dims và group structure
3. Báo cáo reviewer TRƯỚC khi implement SFT
4. Implement SFT injection theo spec từ reviewer
5. Train DRCT baseline + PD-MAE-DRCT song song
6. Evaluate RealSR V3 Canon sau mỗi 50k iters
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
| DRCT injection points không rõ | Đọc code trước, báo reviewer trước khi implement |
| ViT-Small encoder quá nặng | Linear projection 384→DRCT_channels trước SFT |
| Kết quả không beat DRCT | Thử cross-attention thay SFT |
| **[ĐÃ XẢY RA] Data mismatch** | Fix: dùng DRCT's own training pipeline |
| **[ĐÃ XẢY RA] Catastrophic forgetting** | Fix: không train backbone trên LR_all_sub |

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
✅ Stage 1 MAE pretrain    — DONE (200k iters, loss=0.0037)
✅ Stage 2 HR fine-tune    — DONE (100k iters, loss=0.0068)
❌ Stage 3A Real-ESRGAN   — ABANDONED (NIQE 8.08, data mismatch)
🔄 Stage 3B DRCT          — ĐANG BẮT ĐẦU

Lần cập nhật cuối: 25/05/2026
Người cập nhật: Claude (reviewer)
Ghi chú: Pivot từ Real-ESRGAN sang DRCT sau Stage 3A failure.
         Xem PD_MAE_SR_Stage3_Issue_Report.md để biết chi tiết.
         Agent PHẢI đọc Section 12 trong project_context_v2.md
         trước khi implement bất cứ gì cho Stage 3B.
```

---

*Khi cần tư vấn hoặc review, paste file này + kết quả mới nhất vào Claude.*  
*File context đầy đủ hơn: `PD_MAE_SR_project_context.md`*

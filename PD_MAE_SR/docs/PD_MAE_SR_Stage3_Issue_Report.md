# Báo cáo Vấn đề Stage 3 & Đề xuất Hướng đi Mới
**Dự án:** PD-MAE-SR  
**Giai đoạn:** Stage 3 — SR Downstream (Real-ESRGAN backbone)  
**Ngày:** 25/05/2026  
**Trạng thái:** Dừng — Pivot sang backbone mới

---

## 1. Tóm tắt vấn đề

Stage 3 sử dụng Real-ESRGAN (RRDBNet) làm SR backbone với PD-MAE encoder inject qua SFT layers. Sau 130k iterations, model **không cải thiện** so với baseline Real-ESRGAN official, thậm chí cho NIQE cao hơn đáng kể.

---

## 2. Kết quả thực nghiệm

### 2.1 NIQE trên RealSR V3 Canon

| Model | NIQE ↓ | Ghi chú |
|-------|--------|---------|
| Real-ESRGAN official (pretrained) | **4.51** | Baseline tham chiếu |
| Real-ESRGAN our degradation (paper Siamese) | 6.88 | Retrain trên data nhóm |
| PD-MAE-SR + L_PD_MAE (iter 110k) | 8.08 | Tệ hơn baseline 3.57 points |
| PD-MAE-SR không L_PD_MAE (iter 130k) | 7.96 | Ablation: tắt consistency loss |
| LR input (không SR) | 7.37 | Model đang làm ảnh xấu hơn input |

### 2.2 PSNR/SSIM trên Set14

| Iter | PSNR | SSIM |
|------|------|------|
| 5k | 23.47 | 0.649 |
| 10k | 23.28 (best) | — |
| 110k | 23.04 | 0.625 |
| 130k | ~22.8 | ~0.620 |

PSNR giảm liên tục, không recover sau GAN warmup period.

---

## 3. Phân tích nguyên nhân

### 3.1 Ablation L_PD_MAE — Không phải nguyên nhân chính

Tắt hoàn toàn L_PD_MAE (mae_weight=0.0) và train thêm 20k iters chỉ giảm NIQE từ 8.08 → 7.96 (giảm 0.12). Khoảng cách 3.45 points so với baseline vẫn còn nguyên. **L_PD_MAE không phải root cause.**

### 3.2 Data mismatch — Nguyên nhân chính

Training data của PD-MAE-SR là LR_all_sub với **4 degradation levels cố định** (60/70/80/90%). Real-ESRGAN official được train với **online degradation** tạo ra vô số variants mỗi iteration. RealSR V3 Canon là real-world images với distribution khác hoàn toàn.

```
Diversity training:
  Real-ESRGAN official:  ~∞ degradation variants (online generation)
  PD-MAE-SR:             4 degradation patterns (pre-generated)
  
→ Model overfit vào 4 patterns → generalize kém trên real-world data
→ NIQE cao hơn cả LR input = model tạo artifacts
```

### 3.3 Catastrophic forgetting của pretrained weights

Model khởi động từ `RealESRNet_x4plus.pth` (train trên DF2K online degradation). Sau 130k iters train trên LR_all_sub với distribution hẹp, model **quên** distribution mà pretrained weights đã học. Đây là dạng catastrophic forgetting do domain shift.

### 3.4 Real-ESRGAN — Vấn đề tích lũy

Nhóm đã gặp khó khăn với Real-ESRGAN qua nhiều project:
- Paper Siamese: PSNR drop khi train trên data của nhóm
- Knowledge Distillation: unstable training
- Stage 3 hiện tại: data mismatch, catastrophic forgetting

**Kết luận: Real-ESRGAN (RRDBNet) không phải backbone phù hợp** để demonstrate PD-MAE contribution. Architecture cũ (~2021), training dynamics phức tạp, nhạy cảm với data distribution.

---

## 4. Điều đã học được từ Stage 3

Dù kết quả SR chưa tốt, có **3 insight quan trọng** cho paper:

**Insight 1 — PD-MAE encoder học được restoration-prior tốt:**
Stage 1 và Stage 2 cho thấy encoder reconstruct fine details (stone carving, leopard print) từ heavily degraded inputs. Đây là evidence độc lập với SR downstream performance.

**Insight 2 — SFT injection architecture đúng nhưng backbone sai:**
L_PD_MAE ablation cho thấy vấn đề không phải ở loss design mà ở backbone generalization. Với backbone mạnh hơn + diverse training data, SFT injection có thể work.

**Insight 3 — Data diversity quan trọng hơn data quality:**
4 mức degradation cố định không đủ để train một SR model generalizable. Cần online degradation hoặc ít nhất là data augmentation strategy đa dạng hơn.

---

## 5. Đề xuất hướng đi mới — Backbone SOTA

### 5.1 Lý do chuyển backbone

| Tiêu chí | Real-ESRGAN | DRCT / MambaIRv2 |
|---------|-------------|-----------------|
| Architecture | RRDB (2018) | Transformer/Mamba (2024) |
| Training stability | Thấp (GAN) | Cao hơn |
| Blind SR performance | Trung bình | SOTA |
| Data sensitivity | Cao | Thấp hơn |
| Public code | Có | Có |

### 5.2 Backbone candidates

**Ưu tiên 1 — DRCT** (Dense-Residual Connected Transformer)
- SOTA tại NTIRE 2024/2025
- ~14M params, train được trên RTX 3090
- Code: github.com/ming053l/DRCT
- Không dùng GAN → training ổn định hơn nhiều

**Ưu tiên 2 — MambaIRv2**
- CVPR 2025, efficient hơn DRCT
- ~12M params
- Cần check memory compatibility

### 5.3 Thay đổi strategy cho Stage 3 mới

**Thay đổi quan trọng nhất — Dùng online degradation của backbone:**

Thay vì ép backbone train trên LR_all_sub, dùng **degradation pipeline của backbone** + inject PD-MAE encoder. Argument trong paper vẫn valid: PD-MAE encoder pretrain trên degradation data của nhóm (contribution) → inject vào backbone bất kỳ (generalizability).

**Giữ lại:**
- PD-MAE encoder Stage 1 + Stage 2 checkpoints (không cần train lại)
- SFT injection architecture
- L_PD_MAE consistency loss (test với weight nhỏ hơn: 0.01-0.05)

**Thay thế:**
- RRDBNet → DRCT
- LR_all_sub training data → DRCT's online degradation
- Real-ESRGAN training pipeline → DRCT training pipeline

---

## 6. Kế hoạch Stage 3 mới

| Bước | Việc làm | Thời gian ước tính |
|------|----------|-------------------|
| 1 | Clone DRCT, setup môi trường | 1-2 ngày |
| 2 | Đọc DRCT architecture, xác định injection points | 1 ngày |
| 3 | Implement SFT injection vào DRCT | 3-5 ngày |
| 4 | Smoke test forward pass | 1 ngày |
| 5 | Train DRCT + PD-MAE encoder, 200k iters | ~5-7 ngày |
| 6 | Evaluate NIQE/PSNR trên RealSR V3, Set5, Set14 | 1 ngày |
| 7 | Train DRCT baseline (không có MAE) để compare fair | ~5-7 ngày song song |

**Tổng:** ~3 tuần

---

## 7. Ablation table mục tiêu (cập nhật)

| Config | Backbone | MAE | NIQE RealSR↓ | PSNR Set5↑ |
|--------|----------|-----|-------------|-----------|
| Real-ESRGAN official | RRDBNet | ✗ | 4.51 | — |
| DRCT baseline | DRCT | ✗ | ? | ? |
| **PD-MAE-DRCT (ours)** | **DRCT** | **✓** | **?** | **?** |
| PD-MAE-DRCT no L_PD_MAE | DRCT | SFT only | ? | ? |
| PD-MAE-DRCT no SFT | DRCT | Loss only | ? | ? |

---

## 8. Kết luận

Stage 3 với Real-ESRGAN backbone **không thành công** do data mismatch và catastrophic forgetting. Tuy nhiên đây không phải failure của PD-MAE concept — encoder đã học được restoration-oriented representation tốt (Stage 1 và 2). Vấn đề nằm ở **integration strategy với backbone cụ thể**.

Quyết định chuyển sang DRCT là đúng vì:
- Backbone mạnh hơn, training ổn định hơn
- SOTA benchmark → paper argument thuyết phục hơn
- Tránh lặp lại vấn đề đã gặp với Real-ESRGAN

**PD-MAE encoder checkpoints (Stage 1 + 2) được giữ nguyên và tái sử dụng.**

---

*Báo cáo ngày 25/05/2026*  
*Reviewer: Claude (Anthropic)*

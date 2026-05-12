# Review như Reviewer Q1: Siamese-based Real-ESRGAN

---

## Tóm tắt đánh giá tổng thể

**Verdict: Weak Reject / Major Revision** (nếu nộp workshop/tier-2 conference) hoặc **Reject** (nếu nộp tier-1 như CVPR/ICCV/ECCV/NeurIPS).

Phù hợp nhất với: **KSE, RIVF, MAPR, hoặc các IEEE conference vùng** — đây là mức thực tế nhất.

---

## Điểm mạnh

- Ý tưởng kết hợp Siamese + curriculum learning + high-order degradation có tính thực tiễn
- Pipeline degradation có cải tiến rõ ràng so với baseline Real-ESRGAN
- Kết quả NIQE trên RealSR V3 cải thiện ~16.5% là con số đáng chú ý
- Qualitative results (Fig. 4) trực quan và thuyết phục

---

## Điểm yếu — Chi tiết từng phần

### 1. Novelty & Contribution (Yếu nhất)

**Vấn đề nghiêm trọng:** Bài này là extension của [4] (KSE 2025 của chính nhóm tác giả), nhưng không làm rõ ranh giới contribution mới so với [4] là gì. Reviewer sẽ hỏi ngay:

> *"What is the delta over [4]? Is this just an incremental engineering paper?"*

Phần Introduction liệt kê contributions nhưng không có **bullet "Contributions" rõ ràng**. Bài không có đoạn nào dạng:

> *"Compared to our prior work [4], this paper additionally proposes..."*

**Cần sửa:** Thêm explicit contribution list, và viết hẳn 1 đoạn so sánh với [4].

---

### 2. Related Work (Thiếu nghiêm trọng)

Thiếu các nhóm tài liệu quan trọng sau:

**a) Diffusion-based SR** — bài có *nhắc đến* diffusion models trong Introduction nhưng không cite gì cả. Reviewer sẽ hỏi tại sao không so sánh với:
- StableSR (Wang et al., 2023)
- DiffBIR (Lin et al., 2023)
- PASD (Yang et al., 2023)

**b) Siamese Networks** — dùng Siamese làm contribution chính nhưng Related Work không có mục riêng về Siamese Networks. Thiếu các ref cơ bản:
- Bromley et al. (1993) — gốc Siamese
- Koch et al. (2015) — Siamese for few-shot

**c) Curriculum Learning** — cũng là contribution chính nhưng không cite:
- Bengio et al. (2009) *"Curriculum learning"* — **bắt buộc phải có**
- Soviany et al. (2022) survey on curriculum learning

**d) Blind Image Quality Assessment** — dùng NIQE làm metric chính nhưng không cite paper gốc:
- Mittal et al. (2013) *"Making a completely blind image quality analyzer"* — **thiếu ref gốc của NIQE**

**e) Knowledge Distillation cho SR** — chỉ có 2 ref [11][12], thiếu:
- FAKD (He et al., 2020)
- các KD-based SR gần đây hơn

---

### 3. Methodology (Một số điểm mơ hồ)

**a) Loss function không đầy đủ:**
Equation Ltotal có VGG perceptual loss nhưng cite là `[?]` — **đây là lỗi rất nghiêm trọng**, dứt khoát bị reject vì thiếu citation. Cần cite:
- Johnson et al. (2016) *"Perceptual Losses for Real-Time Style Transfer"*
- Simonyan & Zisserman (2014) cho VGG19

**b) Trọng số loss chưa được ablate:** Tại sao `w_pixel = 2.0`, `w_GAN = 0.5`, `λ_kd = 0.15`? Không có ablation study nào justify các hyperparameter này.

**c) "Degradation levels 90%, 80%, 70%, 60%" không được định nghĩa rõ:** 90% quality nghĩa là gì về mặt kỹ thuật? JPEG quality factor? Composite score? Reviewer sẽ hỏi đây.

**d) Curriculum phases không rõ số iteration:** Mỗi phase train bao nhiêu iteration trong tổng 800k? Chỉ nói "progressively" là chưa đủ.

---

### 4. Experiments (Thiếu so sánh)

**a) Thiếu comparison với SOTA quan trọng:**
- SwinIR có trong Fig. 4 (qualitative) nhưng **không có trong Table I** (quantitative) — đây là inconsistency rõ ràng
- BSRGAN cũng vậy — có trong hình nhưng không có trong bảng
- Không so sánh với bất kỳ diffusion-based method nào

**b) Không có ablation study:**
Đây là điểm yếu lớn nhất về mặt experimental. Cần ablation:
- Có/không có curriculum learning
- Có/không có feature distillation loss
- Có/không có camera noise simulation
- Weight-sharing vs. separate networks

**c) PSNR/SSIM của Siamese Phase 2 thấp hơn baseline** trên Set5 (25.96 vs 26.59) và RealSR Canon (25.83 vs 27.56) — nhưng bài không giải thích thỏa đáng sự trade-off này. Reviewer sẽ hỏi tại sao claim "superior" khi distortion metrics thấp hơn.

**d) Không có user study hoặc perceptual study** trong khi claim về "visual realism."

---

### 5. Writing & Presentation

- **Abstract nói NIQE = 3.78** nhưng Table I không có con số này ở đâu — inconsistency
- Fig. 1 chú thích *"adapted from [4]"* — nếu figure gần giống [4] thì đây là vấn đề self-plagiarism cần cẩn thận
- Không có **complexity analysis** (FLOPs, inference time, parameters) dù claim "low computational cost"
- Section IV.B Training Details để `[?]` citation — **không thể nộp bài với lỗi này**

---

## Bảng tóm tắt điểm cần sửa theo mức độ ưu tiên

| Mức độ | Vấn đề |
|--------|--------|
| 🔴 Bắt buộc | Fix `[?]` citation cho VGG/perceptual loss |
| 🔴 Bắt buộc | Thêm SwinIR, BSRGAN vào Table I cho nhất quán |
| 🔴 Bắt buộc | Cite NIQE paper gốc (Mittal et al. 2013) |
| 🔴 Bắt buộc | Cite Curriculum Learning (Bengio et al. 2009) |
| 🟠 Quan trọng | Thêm Ablation Study |
| 🟠 Quan trọng | Làm rõ delta contribution so với [4] |
| 🟠 Quan trọng | Định nghĩa rõ "degradation quality %" |
| 🟠 Quan trọng | Thêm complexity/inference time analysis |
| 🟡 Nên có | So sánh với ít nhất 1 diffusion-based method |
| 🟡 Nên có | Thêm Siamese Networks vào Related Work |
| 🟡 Nên có | Giải thích PSNR/SSIM thấp hơn baseline |
| 🟡 Nên có | Nhất quán con số NIQE 3.78 trong Abstract với Table |

---

## Kết luận

Bài có hướng đi tốt và kết quả thực nghiệm khá thuyết phục về mặt perceptual quality. Tuy nhiên để được accept ở bất kỳ venue nghiêm túc nào, cần giải quyết tối thiểu các lỗi đỏ trước khi nộp. Với các sửa đổi đầy đủ (đặc biệt ablation study và hoàn thiện related work), bài phù hợp với **IEEE RIVF, MAPR, KSE, hoặc các IEEE conference chuyên ngành về signal/image processing** như ICIP hoặc ISCAS.

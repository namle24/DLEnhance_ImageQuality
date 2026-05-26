# Phân tích Kiến trúc DRCT & Chiến lược Tích hợp SFT (Stage 3B)

Tài liệu này ghi lại kết quả phân tích mã nguồn DRCT (`DRCT/drct/archs/DRCT_arch.py`) và cấu hình huấn luyện blind SR (`train_Real_DRCT_SRx4_mse_model.yml`), đồng thời đề xuất giải pháp chèn lớp SFT (Spatial Feature Transform) kết nối với PD-MAE Encoder.

---

## 1. Phân tích chi tiết kiến trúc DRCT

Kiến trúc DRCT bao gồm 3 phần chính hoạt động theo luồng:

1. **Shallow Feature Extraction (Trích xuất nông):**
   * Lớp `self.conv_first = nn.Conv2d(3, embed_dim, 3, 1, 1)` chuyển đổi ảnh đầu vào LR (3 kênh) thành bản đồ đặc trưng có số kênh là `embed_dim`.
2. **Deep Feature Extraction (Trích xuất sâu - Body):**
   * **`self.patch_embed`:** Chuyển đổi bản đồ đặc trưng 2D spatial `[B, embed_dim, H, W]` thành dạng chuỗi 1D token `[B, H*W, embed_dim]`. Vì đây là bài toán SR cổ điển, `patch_size` được thiết lập mặc định là `1` (không làm giảm độ phân giải của đặc trưng).
   * **`self.layers`:** Chuỗi các nhóm **RDG (Dense-Residual Connected Transformer Groups)**.
     * Mỗi nhóm RDG chứa **5 khối Swin Transformer (`SwinTransformerBlock`)** liên kết theo kiểu Dense-Residual Conv: đặc trưng đầu ra của khối trước được concatenate với đầu vào để đưa vào khối tiếp theo sau khi đi qua một lớp tích chập chuyển đổi kênh (`nn.Conv2d`).
   * **`self.norm`:** Lớp `LayerNorm` chuẩn hóa chuỗi token.
   * **`self.patch_unembed`:** Khôi phục chuỗi token 1D `[B, H*W, embed_dim]` trở lại bản đồ đặc trưng 2D spatial `[B, embed_dim, H, W]`.
   * **`self.conv_after_body`:** Phép tích chập $3\times3$ cộng trực tiếp đặc trưng nông (residual connection).
3. **HQ Reconstruction (Tái tạo chất lượng cao):**
   * Gồm chuỗi các phép tích chập `conv_before_upsample`, lớp `Upsample` (sử dụng cơ chế PixelShuffle) và tích chập cuối `conv_last` để chuyển về ảnh màu HR 3 kênh.

---

## 2. Số lượng nhóm (Groups) & Kích thước kênh (Channel Dims)

Dựa trên tệp cấu hình huấn luyện blind SR thực tế (`train_Real_DRCT_SRx4_mse_model.yml`), các tham số kiến trúc được định nghĩa như sau:

| Tham số | Giá trị mặc định trong code | Giá trị thực tế trong cấu hình Blind SR (`Real_DRCT`) |
| :--- | :--- | :--- |
| **Số lượng nhóm RDG** (`depths`) | **4 nhóm** (`depths: [6, 6, 6, 6]`) | **6 nhóm** (`depths: [6, 6, 6, 6, 6, 6]`) |
| **Số kênh đặc trưng Swin** (`embed_dim`) | **96 kênh** | **180 kênh** |
| **Số head chú ý** (`num_heads`) | `[6, 6, 6, 6]` | `[6, 6, 6, 6, 6, 6]` |

* **Đặc trưng PD-MAE Encoder:** Có số kênh cố định là **384** (mặc định của ViT-Small).
* **Đặc trưng DRCT:** Có số kênh hoạt động trong body là **180** (theo cấu hình huấn luyện).
* **Lớp SFT (Spatial Feature Transform):** Nhận đầu vào đặc trưng DRCT (`180` kênh) và đặc trưng MAE (`384` kênh), thực hiện chiếu tuyến tính đặc trưng MAE thông qua Conv $1\times1$ để sinh ra Scale và Shift tương thích với `180` kênh của DRCT.

---

## 3. Chiến lược chèn SFT (SFT Injection Points)

Do đặc trưng bên trong các nhóm RDG có dạng chuỗi 1D token `[B, L, C]` và việc nội suy kích thước (để khớp độ phân giải bản đồ đặc trưng giữa MAE và DRCT) thuận tiện nhất trên không gian 2D spatial `[B, C, H, W]`, cơ chế chèn SFT tối ưu là:
> **Un-embed đặc trưng sang 2D $\rightarrow$ Áp dụng SFT $\rightarrow$ Embed trở lại dạng 1D token.**

```python
x_2d = self.patch_unembed(x, x_size)               # [B, C, H, W]
x_2d = self.sft_layers[str(i)](x_2d, mae_feat)      # Áp dụng SFT (đã nội suy kích thước MAE feat)
x = self.patch_embed(x_2d)                          # [B, L, C]
```

### Các tùy chọn vị trí chèn:

* **Tùy chọn A (Đồng đều giữa-cuối) — [ĐÃ CHỌN]:** Chèn sau các nhóm RDG số **`[1, 3, 5]`** (đối với cấu hình 6 nhóm) hoặc `[1, 2, 3]` (đối với cấu hình 4 nhóm). Cách này giúp dẫn dắt thông tin cấu trúc qua các cấp độ đặc trưng khác nhau mà không làm thay đổi luồng xử lý chính.
* **Tùy chọn B (Huấn luyện phân tán toàn bộ):** Chèn sau tất cả các nhóm RDG `[0, 1, 2, 3, 4, 5]`.
* **Tùy chọn C (Cuối khối trích xuất sâu):** Chèn duy nhất 1 lớp SFT sau khi đi qua toàn bộ body (ngay sau `self.patch_unembed` và trước `self.conv_after_body`).

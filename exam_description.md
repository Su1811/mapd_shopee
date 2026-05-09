# Bài tập nhóm cuối kì: Tối ưu hóa giao hàng đa tác tử thời gian thực

Bài toán mô phỏng hệ thống giao hàng thực tế: một đội $C$ shipper hoạt động đồng thời trên bản đồ lưới $N \times N$, nhận và giao các kiện hàng có trọng lượng, mức ưu tiên và deadline khác nhau. Đơn hàng xuất hiện liên tục theo thời gian với tốc độ biến động — bao gồm các đợt cao điểm (surge) tập trung tại một số khu vực đặc biệt (hotspot) — tạo ra nút cổ chai cục bộ. Nhiệm vụ của nhóm là thiết kế thuật toán phân công và điều phối shipper để **tối đa hóa tổng phần thưởng** trong $T$ bước thời gian, cân bằng giữa giao đúng hạn, xử lý đơn ưu tiên cao và chi phí di chuyển.

> **Thang thời gian:** 1 giờ = 10 đơn vị thời gian &nbsp;|&nbsp; 1 ngày = 240 đơn vị thời gian

### Files được cấp

- `algo.py` — file baseline (Greedy BFS). **Đây là file duy nhất nhóm cần chỉnh sửa.**
- `run_test.py` — grader chính thức, chấm điểm tự động. **Không được sửa.**
- `test_config.txt` — 6 config Phase 1 kèm bản đồ. **Không được sửa.**
- `demo_notebook.ipynb` — Kaggle notebook gọi terminal, không chứa thuật toán.

### Phase 1 — Phát triển và nộp bài

Nhóm nhận `test_config.txt` để phát triển và kiểm tra các thuật toán được yêu cầu:

```bash
python run_test.py --config test_config.txt --out results/ --seed 42
```

Nộp trên Kaggle: **1 version duy nhất**, vi phạm sẽ bị trừ điểm. Code và báo cáo nộp ở Phase 1 là phiên bản chính thức được đánh giá và chấm.

### Phase 2 — Config ẩn

Ba ngày trước deadline Phase 2, ban tổ chức release `test_config_phase2.txt` (8 config, bản đồ lớn đến $N = 100$, nhiều đợt surge). Nhóm dùng đúng `algo.py` đã nộp để chạy, lưu ý **TUYỆT ĐỐI KHÔNG** được sửa:

Thời gian chạy tối đa: **60 phút** tổng. Tắt internet khi chạy (`Kaggle Settings → Internet → Off`).

> **Phase 1:** Các tham số surge (windows, hotspot, amplitude) **không được công bố**. Nhóm tự thiết kế chiến lược ứng phó dựa trên mô tả cơ chế ở mục 1.3.
>
> **Phase 2:** Tham số surge và hotspot **được công bố trong `test_config_final.txt`**. Thời điểm và vị trí xuất hiện từng đơn cụ thể vẫn là ngẫu nhiên theo seed.

---

## 1. Mô tả bài toán

Cho bản đồ dạng lưới $A$ kích thước $N \times N$, trong đó $A[i][j] = 0$ là ô trống và $A[i][j] = 1$ là ô vật cản, với $1 \leq i, j \leq N$.

Tại $t = 0$, có $C$ shipper trên bản đồ. Shipper $i$ có toạ độ $(x_i, y_i)$, tải trọng tối đa $W_{\max}(i)$ và sức chứa $K(i)$ đơn. Không có hai shipper nào đứng cùng ô.

---

### 1.1. Tập hành động

Tại mỗi bước $t$, mỗi shipper thực hiện một cặp hành động $(move, cargo\_op)$:

**$move \in \{S, L, R, U, D\}$** — S: đứng yên; L/R/U/D: di chuyển Tây/Đông/Bắc/Nam một ô.

**$cargo\_op$** — `0`: không làm gì; `1`: nhặt đơn tại ô hiện tại; `2 [id]`: giao đơn `id` đang mang.

Thứ tự trong một bước: **Di chuyển → Nhặt hàng → Giao hàng**.

---

### 1.2. Mô hình đơn hàng

$$g_i = \langle sx_i,\; sy_i,\; ex_i,\; ey_i,\; et_i,\; w_i,\; p_i \rangle$$

| Thuộc tính | Ý nghĩa |
|---|---|
| $sx_i, sy_i$ | Toạ độ điểm lấy hàng |
| $ex_i, ey_i$ | Toạ độ điểm giao hàng |
| $et_i$ | Deadline (đơn vị thời gian) — quá hạn bị phạt |
| $w_i$ | Khối lượng kiện hàng (kg) |
| $p_i \in \{1,2,3\}$ | Mức ưu tiên: 1 = Tiêu chuẩn, 2 = Nhanh, 3 = Hỏa tốc |

---

### 1.3. Mô hình sinh đơn hàng: Surge & Hotspot

Đơn hàng xuất hiện theo **quá trình Poisson không đồng nhất** với tốc độ $\lambda(t)$ thay đổi theo thời gian:

$$\lambda(t) = \begin{cases} \lambda_0 \times (1 + A) & \text{nếu } t \in [t_s,\, t_e] \quad \text{(surge window)} \\ \lambda_0 & \text{ngược lại} \end{cases}$$

| Tham số | Ý nghĩa |
|---|---|
| $\lambda_0 \approx G / T$ | Tốc độ sinh đơn nền (đơn/bước thời gian) |
| $A \geq 0$ | Biên độ surge — hệ số khuếch đại tốc độ trong cao điểm |
| $[t_s,\, t_e]$ | Surge window — khoảng thời gian xảy ra cao điểm |
| Hotspot $(r, c)$ | Tâm khu vực đặc biệt: đơn hàng tập trung mạnh gần đây |

**Cơ chế hotspot:** Trong một surge window, với xác suất 70%, điểm lấy hàng $(sx, sy)$ được chọn ngẫu nhiên trong vùng lân cận Manhattan ≤ 3 quanh một hotspot. Xác suất 30% còn lại vẫn chọn ngẫu nhiên toàn bản đồ. Điều này tạo ra **nút cổ chai cục bộ**: nhiều đơn hàng cùng xuất hiện gần nhau trong thời gian ngắn, buộc shipper phải cạnh tranh tài nguyên di chuyển.

**Ví dụ trực quan:**

```
Bình thường (λ₀ = 0.1):        Trong surge (A = 3.0, λ = 0.4):
  . . . . .                       . . . . .
  . . o . .   ← đơn rải đều      . H H H .   ← đơn tập trung
  . o . . .                       . H ★ H .     quanh hotspot ★
  . . . o .                       . H H H .
  . . . . .                       . . . . .
```

> **Phase 1:** Các tham số $\lambda_0$, $A$, danh sách surge windows và hotspots **không được công bố** trong Phase 1 nhằm khuyến khích sinh viên thiết kế thuật toán có khả năng thích nghi với môi trường động, thay vì hard-code theo cấu hình biết trước. Trong trường hợp không có hai trường này trong config thì sẽ mặc định sẽ lấy Random (đã được cấu hình trong code).
>
> **Phase 2:** Tham số surge và hotspot **được công bố đầy đủ trong phase này.

---

### 1.4. Sức chứa và trọng lượng

$$\sum_{j \in bag(i)} w_j \leq W_{\max}(i) \qquad |bag(i)| \leq K(i)$$

Khi nhặt hàng tại ô có nhiều đơn, ưu tiên: **hỏa tốc > nhanh > tiêu chuẩn > (chỉ số nhỏ hơn)**.

| Hạng mục | Khối lượng $w$ | Chi phí/bước $rc(w)$ | Sức chứa $K$ |
|---|---|---|---|
| Nhẹ | $w \leq 3$ kg | $-0.01$ | 3 đơn |
| Trung bình | $3 < w \leq 10$ kg | $-0.02$ | 2 đơn |
| Nặng | $10 < w \leq 30$ kg | $-0.04$ | 1 đơn |
| Siêu nặng | $w > 30$ kg | $-0.08$ | 1 đơn |

---

### 1.5. Hàm phần thưởng

$$r(i) = \begin{cases} \alpha_p \times r_{base}(i) \times (1 + bonus) & \text{nếu } t_{delivery} \leq et_i \\ \beta_p \times r_{base}(i) \times \max\!\left(0,\; 1 - \dfrac{t_{delivery} - et_i}{T}\right) & \text{nếu } t_{delivery} > et_i \end{cases}$$

$$bonus = \max\!\left(0,\; \frac{et_i - t_{delivery}}{et_i}\right)$$

| Loại dịch vụ | $p$ | $\alpha_p$ | $\beta_p$ |
|---|---|---|---|
| Hỏa tốc | 3 | 3.0 | 0.5 |
| Nhanh | 2 | 2.0 | 0.3 |
| Tiêu chuẩn | 1 | 1.0 | 0.1 |

Phần thưởng cơ bản $r_{base}(i) = 10 \times f_{weight}$:

| Trọng lượng $w_i$ | $f_{weight}$ | $r_{base}$ |
|---|---|---|
| $w \leq 0.2$ kg | 0.4 | 4 |
| $0.2 < w \leq 3$ kg | 1.0 | 10 |
| $3 < w \leq 10$ kg | 1.5 | 15 |
| $10 < w \leq 30$ kg | 2.0 | 20 |
| $w > 30$ kg | 3.0 | 30 |

---

### 1.6. Chi phí di chuyển

$$rc(i, t) = -0.01 \times \left(1 + \gamma \cdot \frac{W_{carried}(i,t)}{W_{\max}(i)}\right) \qquad [\gamma = 1.0]$$

Chỉ tính khi shipper di chuyển (`L`/`R`/`U`/`D`). Đứng yên (`S`) không mất chi phí.

---

### 1.7. Hàm mục tiêu

$$\max \quad \sum_i \left[ \sum_{j \text{ giao bởi } i} r(j) \;+\; \sum_t rc(i, t) \right]$$

---

### 1.8. Các ràng buộc vận hành

- **Va chạm:** Shipper có chỉ số nhỏ hơn được ưu tiên giữ ô khi tranh chấp.
- **Thứ tự:** Di chuyển → Nhặt hàng → Giao hàng trong mỗi bước.
- Shipper không được ra ngoài bản đồ hoặc vào ô vật cản.
- Cả $W_{\max}$ và $K(i)$ phải được thỏa mãn mọi lúc.

---

## 2. Các phương pháp cần cài đặt

Yêu cầu với **mỗi** phương pháp:
- Trình bày độ phức tạp thời gian và không gian.
- Phân tích mức độ tối ưu (optimal / near-optimal / heuristic) và điều kiện đảm bảo.
- So sánh kết quả định lượng trên các config Phase 1.

### Bắt buộc — 5 điểm/phương pháp

- Greedy BFS
- VRP + OR-Tools

### Nâng cao — 2.5 điểm/phương pháp

- Ant Colony Optimization (ACO)
- Multi-Agent Pickup and Delivery với Conflict-Based Search (MAPD-CBS)

## 5. Cách nộp bài

```
submission/
├── algo.py                ← thuật toán của nhóm (file duy nhất được sửa)
├── run_test.py            ← KHÔNG SỬA
├── test_config.txt        ← KHÔNG SỬA
├── demo_notebook.ipynb    ← notebook Kaggle submit code
└── report.pdf             ← báo cáo kỹ thuật
```

**Quy tắc Kaggle notebook:**
- Share đúng 1 version.
- Notebook chạy hoàn toàn qua lệnh terminal (`%%bash` cells), không chứa thuật toán.
- Tắt internet khi chạy (`Kaggle Settings → Internet → Off`).
- Seed cố định `--seed 42`.

---

## 6. Thang điểm

| Hạng mục | Điểm | Mô tả |
|---|---|---|
| Greedy BFS | 5 | Cài đặt đúng, chạy được trên tất cả config |
| VRP + OR-Tools | 5 | Cài đặt đúng, chạy được trên tất cả config |
| ACO | 2.5 | Kết quả tốt hơn Greedy BFS, có phân tích |
| MAPD-CBS | 2.5 | Cài đặt đúng, xử lý xung đột đa tác tử |
| Báo cáo kỹ thuật | 5 | Theo yêu cầu mục 6.1 |
| Ranking (Phase 2) | 10 | Nhóm cao nhất = 10đ, các nhóm khác tỉ lệ tuyến tính |
| Vấn đáp | 20 | Từng thành viên trình bày và trả lời câu hỏi |
| **Tổng** | **50** | |

### 6.1. Yêu cầu báo cáo kỹ thuật (5 điểm)

- Ghi rõ thành viên (tối đa 3) và phân công đóng góp.
- Mô tả từng thuật toán: nguyên lý, độ phức tạp thời gian/không gian, mức độ tối ưu.
- Bảng so sánh kết quả định lượng (net reward, % đơn đúng hạn, thời gian chạy) trên từng config Phase 1.
- Phân tích trade-off giữa các phương pháp.
- *(Nâng cao)* Mô tả chiến lược ứng phó surge và hotspot: nhóm phát hiện/xử lý tình huống cao điểm như thế nào?

### 6.2. Điểm ranking (10 điểm)

Dựa trên tổng net reward của `run_test.py` với config Phase 2. Nhóm cao nhất = 10 điểm, các nhóm còn lại tỉ lệ tuyến tính. Điều kiện: notebook chạy lại được độc lập trong 60 phút.

---

*Nhóm tối đa 3 thành viên. Chúc các nhóm thực hiện tốt!*

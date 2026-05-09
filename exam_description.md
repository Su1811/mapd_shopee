# Bài tập lớn: Tối ưu hóa giao hàng đa tác tử trên lưới 2D

## 0. Mục tiêu

Bài toán mô phỏng một hệ thống giao hàng trong đó nhiều shipper hoạt động đồng thời trên bản đồ lưới `N × N`. Mỗi shipper cần nhận và giao các đơn hàng có trọng lượng, mức ưu tiên và deadline khác nhau. Đơn hàng xuất hiện theo thời gian, có thể tăng mạnh trong các khung giờ cao điểm (*surge*) và tập trung quanh một số khu vực nóng (*hotspot*).

Nhiệm vụ của nhóm là thiết kế thuật toán trong `algo.py` để điều phối shipper sao cho **tối đa hóa tổng net reward** trong giới hạn thời gian mô phỏng.

> Thang thời gian: `1 giờ = 10 đơn vị thời gian`, `1 ngày = 240 đơn vị thời gian`.

## 1. Bộ file được cung cấp

```text
submission/
├── algo.py                  # File sinh viên được sửa và nộp
├── env.py                   # Môi trường, setup, load config, sinh đơn; KHÔNG SỬA
├── run_test.py              # Grader chính thức; KHÔNG SỬA
├── test_config.txt          # Config Phase 1 công bố; KHÔNG SỬA
├── test_config_final.txt    # Config Phase 2 ẩn/release sau
└── demo_notebook.ipynb      # Notebook Kaggle chỉ gọi terminal
```

## 2. Cách chạy

Chạy tất cả thuật toán bằng grader chính thức:

```bash
python run_test.py --config test_config.txt --out results/ --seed 42
```

Giới hạn chạy chính thức: **60 phút tổng cho tất cả config**.

## 3. Cấu trúc config

```text
[CONFIG]
name = C1
N = 5
C = 2
G = 15
T = 240
K_max = 3 3
W_max = 20.0 20.0
lambda0 = 0.08
surge_amplitude = 3.0
surge_windows = ss1 se1 ss2 se2
hotspots = hy1 hx1 hy2 hx2
[MAP]
1 1 1 1 1
1 0 0 0 1
1 0 0 0 1
1 0 0 0 1
1 1 1 1 1
[END]
```

| Trường | Ý nghĩa |
|---|---|
| `name` | Tên config |
| `N` | Kích thước bản đồ `N × N` |
| `C` | Số shipper |
| `G` | Số đơn hàng tối đa |
| `T` | Số bước thời gian mô phỏng |
| `K_max` | Sức chứa tối đa theo từng shipper |
| `W_max` | Tải trọng tối đa theo từng shipper |
| `surge_windows` | Danh sách cặp `start end` của các khung giờ cao điểm |
| `hotspots` | Danh sách cặp tọa độ `row col` của hotspot |
| `[MAP]` | Ma trận bản đồ, `0` là ô trống, `1` là vật cản |

`surge_windows = ss1 se1 ss2 se2` nghĩa là có hai khung cao điểm `[ss1, se1]` và `[ss2, se2]`. `hotspots = hy1 hx1 hy2 hx2` nghĩa là có hai hotspot tại `(hy1, hx1)` và `(hy2, hx2)`. Trong trường hợp không có hai trường này trong config thì sẽ mặc định sẽ lấy Random (đã được cấu hình trong code).

## 4. Mô hình bài toán

Bản đồ là lưới `N × N`, trong đó `0` là ô trống và `1` là vật cản. Shipper có thể chọn một trong năm hành động di chuyển: `S` đứng yên, `U` lên, `D` xuống, `L` trái, `R` phải.

Mỗi đơn hàng có dạng:

```text
g_i = <sx, sy, ex, ey, et, w, p>
```

| Thuộc tính | Ý nghĩa |
|---|---|
| `sx, sy` | Vị trí lấy hàng |
| `ex, ey` | Vị trí giao hàng |
| `et` | Deadline |
| `w` | Trọng lượng đơn hàng |
| `p` | Mức ưu tiên: `1 = tiêu chuẩn`, `2 = nhanh`, `3 = hỏa tốc` |

Ràng buộc sức chứa với shipper `i`:

```text
số đơn đang mang ≤ K_max[i]
tổng trọng lượng đang mang ≤ W_max[i]
```

Nếu shipper đã đầy túi hoặc vượt tải, shipper không được nhặt thêm đơn.

## 5. Mô hình sinh đơn hàng

Đơn hàng được sinh trong `env.py` theo quá trình Poisson không đồng nhất. Tốc độ sinh đơn tại thời điểm `t`:

```text
lambda(t) = lambda0 × (1 + surge_amplitude)   nếu t nằm trong surge window
lambda(t) = lambda0                           nếu ngược lại
```

Trong surge window, nếu có hotspot, điểm lấy hàng có xác suất cao tập trung quanh hotspot: `70%` chọn điểm lấy hàng trong bán kính Manhattan `≤ 3` quanh một hotspot, `30%` chọn ngẫu nhiên toàn bản đồ. Điều này tạo tình huống giống thực tế: trong giờ cao điểm, nhiều đơn xuất hiện dày đặc quanh một khu vực, làm tăng cạnh tranh tài nguyên shipper.

## 6. Phase 1 và Phase 2

### Phase 1

Sinh viên nhận `test_config.txt` để phát triển thuật toán. Một số tham số surge/hotspot có thể không xuất hiện trực tiếp trong config công bố nhằm khuyến khích sinh viên thiết kế thuật toán có khả năng thích nghi với môi trường động, thay vì hard-code theo cấu hình biết trước. Điều này giúp đánh giá tốt hơn khả năng quan sát trạng thái mô phỏng, nhận diện dấu hiệu cao điểm qua backlog, mật độ đơn và tỷ lệ trễ, cũng như xây dựng chiến lược phân công linh hoạt.

Sinh viên nộp:

- `algo.py`;
- Kaggle notebook dùng để chạy lại bài;
- báo cáo kỹ thuật `report.pdf`.

Không cần đưa notebook vào trong nội dung báo cáo.

### Phase 2

Giảng viên release `test_config_phase2.txt` trước deadline Phase 2. File này sẽ cung cấp trực tiếp `surge_windows`, `hotspots`. Sinh viên dùng đúng `algo.py` đã nộp để chạy lại trên Kaggle.

## 7. Hàm thưởng và chi phí

Phần thưởng cơ bản theo trọng lượng:

| Trọng lượng | `r_base` |
|---|---:|
| `w ≤ 0.2` | `4` |
| `0.2 < w ≤ 3` | `10` |
| `3 < w ≤ 10` | `15` |
| `10 < w ≤ 30` | `20` |
| `w > 30` | `30` |

Hệ số ưu tiên:

| Mức ưu tiên | Ý nghĩa | `alpha_p` đúng hạn | `beta_p` trễ hạn |
|---|---|---:|---:|
| `1` | Tiêu chuẩn | `1.0` | `0.1` |
| `2` | Nhanh | `2.0` | `0.3` |
| `3` | Hỏa tốc | `3.0` | `0.5` |

Nếu giao đúng hạn:

```text
reward = alpha_p × r_base × (1 + bonus)
bonus = max(0, (deadline - delivery_time) / deadline)
```

Nếu giao trễ:

```text
reward = beta_p × r_base × max(0, 1 - (delivery_time - deadline) / T)
```

Mỗi bước di chuyển thật sự `L/R/U/D` bị trừ:

```text
move_cost = -0.01 × (1 + W_carried / W_max)
```

Đứng yên `S` không mất chi phí.

## 8. Hàm mục tiêu

Thuật toán cần tối đa hóa:

```text
net_reward = total_delivery_reward + total_move_cost
```

Kết quả chính thức lấy từ `run_test.py`:

```text
TỔNG ĐIỂM RANKING = tổng net_reward trên toàn bộ config
```

## 9. Yêu cầu thuật toán và báo cáo

Tối thiểu, nhóm cần cài đặt và phân tích Greedy BFS hoặc cải tiến tương đương, VRP/OR-Tools hoặc heuristic gom cụm tương đương nếu môi trường Kaggle cho phép, và các chiến lược nâng cao nếu muốn tăng điểm ranking như ưu tiên đơn hỏa tốc, gom đơn theo cụm không gian, điều phối shipper đến gần hotspot trước surge, cache đường đi A*, xử lý đơn nặng bằng shipper có `W_max` lớn, tránh nhiều shipper tranh cùng một đơn.

Với mỗi phương pháp trong báo cáo, cần nêu ý tưởng, giả mã hoặc mô tả luồng xử lý, độ phức tạp thời gian, độ phức tạp bộ nhớ, mức độ tối ưu, và bảng kết quả trên Phase 1.

## 10. Quy định Kaggle notebook

Notebook chỉ dùng để chạy lại code, không chứa thuật toán chính. Ví dụ cell:

```bash
%%bash
python run_test.py --config test_config.txt --out results --seed 42
python run_test.py --config test_config_final.txt --out results_final --seed 42
```

Yêu cầu: dùng seed cố định `42`, không sửa `env.py`, `run_test.py`, config trong lúc chấm, và chỉ chỉnh sửa đúng file thuật toán `algo.py` theo quy định đề bài.

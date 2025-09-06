# 🧠 2048 Bot – Advanced Corner Strategy

Bot tự động chơi **2048** sử dụng thuật toán **Expectimax**, chiến lược corner nâng cao, phát hiện trap, và tối ưu endgame.  
Được thiết kế để đạt tới **tile 65536 (2^16)** – thử thách tối thượng 🎯.

---

## 🚀 Tính năng nổi bật
- **Corner Strategy**: Giữ tile lớn nhất ở một góc cố định (TL, TR, BL, BR).
- **Dynamic Depth**: Điều chỉnh độ sâu tìm kiếm tùy theo tình huống.
- **Trap Detection**: Nhận diện và tránh các mẫu dễ bị kẹt.
- **Endgame Optimization**: Chiến lược đặc biệt khi bảng gần đầy.
- **Auto-learning Palette**: Tự học màu sắc mới từ màn hình game.
- **Hỗ trợ nhiều phương thức nhập**: `pyautogui`, `pydirectinput`, `keyboard`, `adb`.

---

## 📦 Yêu cầu hệ thống
- Python **3.8+**
- Các thư viện:
  ```bash
  pip install numpy opencv-python pyautogui keyboard pydirectinput
(Tùy chọn) Android Debug Bridge (ADB) nếu điều khiển trên thiết bị Android.

## ⚙️ Cách chạy

Clone repository:
```bash
git clone https://github.com/your-username/2048_bot.git
cd 2048_bot
```

Chạy bot:

```bash
python 2048_bot_corner_blockers.py --autoplay --corner tr --depth 4
```

Chọn vùng game khi được yêu cầu (kéo chuột qua góc trên-trái và dưới-phải và bấm Enter trên bàn phím).

| Tham số                    | Mặc định   | Mô tả                                                             |
| -------------------------- | ---------- | ----------------------------------------------------------------- |
| `--autoplay`               | `True`     | Tự động chơi game                                                 |
| `--input`                  | `keys`     | Phương thức nhập: `keys`, `swipe`                                 |
| `--driver`                 | `keyboard` | Trình điều khiển: `pyautogui`, `pydirectinput`, `keyboard`, `adb` |
| `--adb-serial`             | `None`     | Serial thiết bị Android                                           |
| `--swipe-ratio`            | `0.6`      | Tỉ lệ khoảng cách swipe (0.2–1.0)                                 |
| `--depth`                  | `4`        | Độ sâu tìm kiếm Expectimax cơ bản                                 |
| `--goal`                   | `0`        | Tile mục tiêu (0 = không giới hạn)                                |
| `--corner`                 | `tr`       | Chiến lược góc: `tl`, `tr`, `bl`, `br`                            |
| `--disable-trap-detection` | -          | Tắt phát hiện trap                                                |

## 📊 Log & Phân tích

Mỗi lần chạy, bot tạo thư mục runs/YYYYMMDD-HHMMSS/ chứa:

run.log: Nhật ký chi tiết từng bước.

color_palette.json: Dữ liệu màu sắc tile đã học.

## 🛠️ Mẹo sử dụng

Nếu gặp màu chưa biết, nhập giá trị thủ công (hex=value) khi bot hỏi.

Nên dùng pydirectinput thay vì pyautogui để giảm độ trễ.

Khi muốn ghi đè control hoàn toàn, dùng --driver adb.

## 🏆 Mục tiêu

Target: Đạt tile 65536 với chiến lược corner tối ưu.
Power-ups: 🔨 Hammer (2) · 💣 Bomb (1) · 🪄 Wand (3)

## 📜 Giấy phép

Dự án phát hành theo giấy phép MIT.
Bạn có thể tự do chỉnh sửa và sử dụng trong nghiên cứu hoặc giải trí.

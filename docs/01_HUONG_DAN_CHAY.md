# Hướng dẫn chạy — từng bước, có kỳ vọng đầu ra

Làm tuần tự. Mỗi bước có phần **"phải thấy gì"** để bạn tự đối chiếu. Bước nào lệch thì dừng, đừng chạy tiếp.

Toàn bộ lệnh dưới đây dùng **PowerShell trên Windows**. Nếu bạn dùng macOS/Linux, đổi `.\.venv\Scripts\python.exe` thành `.venv/bin/python`.

---

# PHẦN A — Giải nén và đặt đúng chỗ

## A1. Xem bạn đang có gì

```powershell
cd E:\Project\blackjack-mdp
dir
```

Nếu đây là lần đầu và thư mục chưa tồn tại, tạo nó:

```powershell
mkdir E:\Project\blackjack-mdp
cd E:\Project\blackjack-mdp
```

## A2. Giải nén

```powershell
Expand-Archive -Path "$HOME\Downloads\blackjack-complete.zip" -DestinationPath "$HOME\Downloads\bjc" -Force
dir "$HOME\Downloads\bjc\bj2"
```

**Phải thấy:** `blackjack`, `tests`, `docs`, `figures`, `main.py`, `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`

> Chú ý có thư mục con `bj2` bên trong — đó mới là project.

## A3. Xoá code cũ (nếu có), giữ `.venv`

```powershell
cd E:\Project\blackjack-mdp
pwd
```

**Xác nhận `pwd` in ra đúng `E:\Project\blackjack-mdp` trước khi chạy lệnh xoá.**

```powershell
Remove-Item -Recurse -Force .\blackjack -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\tests -ErrorAction SilentlyContinue
Remove-Item -Force .\main.py -ErrorAction SilentlyContinue
```

## A4. Copy code mới vào

```powershell
$src = "$HOME\Downloads\bjc\bj2"
Copy-Item -Recurse "$src\blackjack" .\blackjack
Copy-Item -Recurse "$src\tests" .\tests
Copy-Item -Recurse "$src\docs" .\docs -Force
Copy-Item -Recurse "$src\figures" .\figures -Force
Copy-Item "$src\main.py" .\main.py
Copy-Item "$src\README.md" .\README.md -Force
Copy-Item "$src\requirements.txt" .\requirements.txt -Force
Copy-Item "$src\pytest.ini" .\pytest.ini -Force
Copy-Item "$src\.gitignore" .\.gitignore -Force
```

Kiểm tra:

```powershell
(Get-ChildItem .\blackjack\*.py).Count
```

**Phải thấy:** `14`

## A5. Tạo môi trường ảo (nếu chưa có)

```powershell
python -m venv .venv
```

Kiểm tra nó tạo thật:

```powershell
dir .venv\Scripts
```

**Phải thấy:** `python.exe`, `pip.exe`, `Activate.ps1`

> Nếu `python` không nhận, thử `py -m venv .venv`. Nếu vẫn lỗi, Python chưa có trong PATH — cài lại Python và tick "Add Python to PATH".

## A6. Cài thư viện

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Phải thấy:** dòng cuối `Successfully installed ...` hoặc `Requirement already satisfied`

> **Tại sao tôi luôn dùng `.\.venv\Scripts\python.exe` thay vì activate?** Vì nó chạy được mọi lúc, không phụ thuộc execution policy của Windows. Nếu bạn muốn activate cho gọn: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi `.\.venv\Scripts\Activate.ps1`.

✅ **Checkpoint A:** `(Get-ChildItem .\blackjack\*.py).Count` ra 14, và pip cài xong không lỗi.

---

# PHẦN B — Chạy kiểm tra

## B1. Toàn bộ test (~35 giây)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**Phải thấy:**
```
117 passed in 35.xx s
```

Nếu ra `failed`, dừng lại. Chạy `pytest -v` để xem test nào đỏ.

## B2. Chỉ test nhanh (~8 giây)

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not slow"
```

**Phải thấy:** `116 passed, 1 deselected`

Dùng lệnh này khi bạn đang sửa code và muốn phản hồi nhanh.

## B3. Xem tên từng test — làm ít nhất một lần

```powershell
.\.venv\Scripts\python.exe -m pytest -v -m "not slow"
```

Đọc chậm danh sách này. **Tên test chính là danh sách những gì dự án tuyên bố.** Chỗ nào bạn không hiểu tại sao test đó tồn tại, đó là chỗ cần đọc lại code.

---

## B4. Phase 1 — nghiệm chính xác (~2 giây)

```powershell
.\.venv\Scripts\python.exe main.py dp
```

**Phải thấy chính xác:**
```
value iteration converged in 13 sweeps
                            computed   published
EV, hit/stand only          -0.02421    -0.02421
EV, doubling allowed        -0.01087    -0.01087
value of the double           1.334%
```

Cột `published` là giá trị đã công bố cho bộ luật này. Khớp đến 5 chữ số nghĩa là mọi chi tiết luật đều cài đúng.

## B5. Bảng chiến thuật (~2 giây)

```powershell
.\.venv\Scripts\python.exe main.py dp --chart --double
```

**Phải thấy** hai bảng `HARD TOTALS` và `SOFT TOTALS` với các ô `S` / `H` / `D`.

**Việc nên làm:** mở một basic strategy chart trên mạng (ví dụ Wizard of Odds, chọn 6 bộ, S17) và so từng ô. Bảng hard sẽ khớp 100%. Bảng soft lệch 2 ô — xem giải thích ở `04_KET_QUA_VA_KIEM_CHUNG.md` mục "Hai ô soft".

## B6. Phase 2 — Q-Learning (~70 giây)

```powershell
.\.venv\Scripts\python.exe main.py ql --episodes 5000000
```

**Phải thấy (gần đúng, seed 42):**
```
mean squared value error        6.47e-05
largest value error             6.29e-02
matching decisions                98.5%
cost of the mismatches           2.47 bps

3 cells disagree, value gap between actions:
  soft 18 vs A: 0.0007
  soft 18 vs 2: 0.1150
  hard 12 vs 6: 0.0106
```

Nếu bạn để `--episodes` nhỏ hơn, sai số sẽ lớn hơn — đó là bình thường.

## B7. Phase 3 — edge theo true count (~60 giây với 3 triệu ván)

```powershell
.\.venv\Scripts\python.exe main.py count --hands 3000000
```

**Phải thấy** bảng 8 dòng, edge **tăng dần** từ khoảng −3.4% lên +0.9%, và:
```
overall finite-shoe edge      -1.0742 %
infinite-deck EV (Phase 1)    -1.0867 %
difference                     1.25 bps
```

Đây là **kiểm chứng chéo quan trọng nhất** của dự án: hai con số đến từ hai phép tính hoàn toàn khác nhau mà khớp trong 1 sai số chuẩn.

Muốn nhanh hơn để thử: `--hands 500000` (kết quả nhiễu hơn, khoảng tin cậy rộng hơn).

## B8. Phase 4 — Kelly và rủi ro (~3 phút)

```powershell
.\.venv\Scripts\python.exe main.py risk --paths 400
```

**Phải thấy** bảng 4 chiến lược. Điểm cần kiểm:
- `full Kelly` có **MDD và ruin cao hơn** `half Kelly` → đúng như lý thuyết
- Dòng `sit out` có vốn cuối **cao hơn** dòng `must bet` → phát hiện chính của Phase 4

## B9. Quét ω (~3 phút)

```powershell
.\.venv\Scripts\python.exe main.py omega
```

**Phải thấy** MSE giảm khi ω tăng, **nhưng phẳng lại ở 0.9–1.0** (khoảng 2.2e−4 và 2.3e−4). Đây không phải lỗi — xem `05_BON_LOI.md`.

## B10. Sinh lại toàn bộ hình (~5 phút)

```powershell
.\.venv\Scripts\python.exe main.py figures
```

Rồi kiểm:

```powershell
dir .\figures
```

**Phải thấy 5 file:** `convergence.png`, `policy.png`, `omega.png`, `edge_by_count.png`, `bankroll.png`

Mở xem:

```powershell
start .\figures\edge_by_count.png
```

✅ **Checkpoint B:** cả 7 lệnh CLI chạy không lỗi, số liệu khớp.

---

# PHẦN C — Xử lý sự cố thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `ModuleNotFoundError: No module named 'blackjack'` | Đang không ở thư mục gốc project | `cd E:\Project\blackjack-mdp` |
| `ModuleNotFoundError: No module named 'numpy'` | Chưa cài thư viện, hoặc dùng sai python | Chạy lại A6, và luôn dùng `.\.venv\Scripts\python.exe` |
| `running scripts is disabled` | Windows chặn `.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `'&&' is not a valid statement separator` | PowerShell 5.1 không hỗ trợ `&&` | Chạy từng lệnh một dòng |
| `The term '.venv\Scripts\activate' is not recognized` | Thiếu `.\` ở đầu | `.\.venv\Scripts\Activate.ps1` |
| `FileNotFoundError: 'figures/...'` | Thư mục `figures` chưa có | `mkdir figures` rồi chạy lại |
| Test đỏ ở `test_...converges...` | Đó là test slow, cần 1 triệu ván | Bình thường nếu chỉ chậm; nếu **đỏ** thì gửi output |

---

# PHẦN D — Việc làm hàng ngày từ giờ

Sau khi Checkpoint B xanh, đây là nhịp làm việc:

**Mỗi lần sửa code:**
```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not slow"
```
8 giây. Làm thường xuyên.

**Trước khi commit:**
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
35 giây, chạy cả test hội tụ.

**Sau khi đổi bất cứ thứ gì liên quan đến thuật toán học:**
```powershell
.\.venv\Scripts\python.exe main.py ql --episodes 5000000
.\.venv\Scripts\python.exe main.py figures
```
Rồi **cập nhật số trong README** nếu chúng đổi. Đây là bước dễ quên nhất, và là nguồn của lỗi "README nói một đằng code làm một nẻo".

---

# Tiếp theo

Chạy được rồi thì sang **02_HIEU_CODE.md** để hiểu từng file làm gì.

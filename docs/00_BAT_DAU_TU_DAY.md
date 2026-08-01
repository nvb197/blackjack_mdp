# Bắt đầu từ đây

Tài liệu này là bản đồ. Đọc nó trước, rồi đi tới tài liệu bạn cần.

---

## 1. Dự án này là gì, trong một câu

> Blackjack với bộ bài vô hạn là bài toán quyết định **giải chính xác được**. Tôi giải nó bằng quy hoạch động để có đáp án chuẩn, dùng đáp án đó kiểm tra một thuật toán học tăng cường không biết luật chơi, rồi mở rộng sang bộ bài hữu hạn để đo lợi thế của việc đếm bài và cách định cỡ tiền cược trên lợi thế đó.

Điểm mấu chốt không phải "tôi làm AI chơi blackjack". Điểm mấu chốt là **có đáp án đúng để đối chiếu**, và **đo sai lệch bằng đơn vị có nghĩa về tiền**.

---

## 2. Bốn giai đoạn

| Phase | Làm gì | Kết quả chính |
|---|---|---|
| **1** | Giải chính xác bằng Value Iteration | EV = **−1.087%**, khớp số đã công bố đến 5 chữ số |
| **2** | Q-Learning không biết luật, so với Phase 1 | khớp **98.5%**, sai lệch tốn **2.47 bps** |
| **3** | Bộ bài hữu hạn 6 bộ + đếm bài Hi-Lo | edge tăng **đơn điệu** theo count, đảo dấu ở TC ≈ +2 |
| **4** | Kelly + đo rủi ro (VaR, CVaR, drawdown, ruin) | cược theo Kelly đáng **+61.45** (t=12.80) |

Cộng thêm một phần quan trọng không nằm trong 4 phase: **4 lỗi được tìm ra qua soát xét**, không cái nào bị test bắt.

---

## 3. Bộ tài liệu — đọc theo thứ tự này

| File | Nội dung | Khi nào đọc |
|---|---|---|
| **00_BAT_DAU_TU_DAY.md** | Bản đồ (bạn đang ở đây) | Đầu tiên |
| **01_HUONG_DAN_CHAY.md** | Từng bước: giải nén, cài, chạy, kỳ vọng thấy gì | **Ngay sau file này** |
| **02_HIEU_CODE.md** | Giải thích từng file code, cho người mới | Sau khi chạy được |
| **03_TOAN_HOC.md** | Toán **đầy đủ, có dẫn chứng minh**: MDP, Bellman, Value Iteration, Q-Learning, hội tụ, đếm bài, thống kê, Kelly, VaR/CVaR — 10 phần, ~800 dòng | Song song với 02, đọc kỹ |
| **04_KET_QUA_VA_KIEM_CHUNG.md** | Mọi con số, nghĩa là gì, kiểm chứng thế nào | Sau khi hiểu code |
| **05_BON_LOI.md** | Bốn lỗi đã sửa — phần giá trị nhất | Trước khi phỏng vấn |
| **06_GITHUB_VA_LO_TRINH.md** | Đưa lên GitHub, chuẩn bị phỏng vấn, việc tiếp theo | Khi sẵn sàng công khai |
| **07_THUAT_NGU_VA_KHAI_NIEM.md** | Từ điển tra cứu — mọi khái niệm lập trình, xác suất, RL, tài chính dùng trong dự án | Tra cứu khi gặp từ lạ ở bất kỳ file nào |

---

## 4. Trạng thái hiện tại

```
117 test xanh
5 hình đã sinh
6 lệnh CLI chạy được
Mọi con số trong README khớp code sống (đã đối chiếu 11/11)
```

**Chưa nên làm:** đưa link vào CV, hoặc bật repo Public — cho tới khi bạn đọc xong 02 và 03 và tự giải thích được code.

---

## 5. Ba con số phải thuộc lòng

| | |
|---|---|
| **−1.087%** | EV khi chơi hoàn hảo, bộ bài vô hạn, có Double |
| **2.47 bps** | Chi phí sai lệch của Q-Learning, so với house edge 242 bps |
| **+0.890%** | Edge ở true count ≥ +3 — nơi lợi thế **đảo dấu** |

Nếu ai hỏi nhanh về dự án, ba số này cùng một câu ở Mục 1 là đủ cho 30 giây đầu.

---

## 6. Cấu trúc thư mục

```
blackjack-mdp/
├── blackjack/              14 file .py — toàn bộ logic
│   ├── rules.py            luật bài, số học tay bài
│   ├── hand.py             luật chơi dùng chung cho 2 môi trường
│   ├── env.py              mô phỏng bộ bài vô hạn
│   ├── shoe.py             shoe 6 bộ, rút không hoàn lại
│   ├── finite_env.py       mô phỏng bộ bài hữu hạn + count
│   ├── counting.py         Hi-Lo, true count, tracker chống look-ahead
│   ├── dp.py               nghiệm chính xác (Value Iteration)
│   ├── qlearning.py        agent học, bộ bài vô hạn
│   ├── qlearning_count.py  agent học, có count trong state
│   ├── sizing.py           Kelly
│   ├── risk.py             VaR, CVaR, drawdown, risk of ruin
│   ├── simulate.py         chơi ván, đo edge, mô phỏng bankroll
│   └── plots.py            5 hình
├── tests/                  117 test
├── docs/                   bộ tài liệu này
├── figures/                5 file .png
├── main.py                 dòng lệnh
└── README.md               bản trình bày cho người ngoài
```

---

## 7. Một điều cần biết trước khi đi tiếp

Dự án này **không** chứng minh bạn thắng được sòng bài. EV vẫn âm trong hầu hết cấu hình. Giá trị của nó là:

1. Có **đáp án chính xác** làm chuẩn — rất hiếm trong bài toán thực tế
2. Kiểm chứng **ba đường độc lập** — đại số, mô phỏng, và học tăng cường
3. Đo sai số bằng **basis point**, không bằng "phần trăm ô đúng"
4. Có **dấu vết tự soát xét** — 4 lỗi được ghi lại thay vì xoá

Điểm 4 là điểm mạnh nhất, và cũng là điểm khó tin nhất nếu bạn không hiểu nó. Đọc **05_BON_LOI.md** kỹ.

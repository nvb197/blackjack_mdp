# Mọi kết quả — nghĩa là gì, và kiểm chứng thế nào

Mọi con số dưới đây đã được đối chiếu với code đang chạy.

---

# PHẦN 1 — Phase 1: nghiệm chính xác

## 1.1 Kết quả

```
value iteration converged in 13 sweeps
EV, hit/stand only    -0.02421   (published -0.02421)
EV, doubling allowed  -0.01087   (published -0.01087)
value of the double     1.334%
```

## 1.2 Nghĩa là gì

**−0.02421** = cứ cược 1 đơn vị, bạn lỗ trung bình 0.02421 đơn vị, tức **lỗ 2.421 đơn vị trên 100 ván**, với điều kiện bạn chơi **hoàn hảo** và chỉ được Hit/Stand.

Đây không phải giá trị một ván cụ thể — nó là trung bình có trọng số trên **toàn bộ** thế bài mở đầu có thể xảy ra, mỗi thế nhân xác suất xuất hiện, bao gồm cả blackjack tự nhiên hai bên.

**value of the double = 1.334%** — chỉ riêng việc *được phép* nhân đôi cược (và dùng đúng lúc) xoá đi hơn **một nửa** lợi thế nhà cái. Ấn tượng ở chỗ bạn chỉ double khoảng 9–10% số ván.

## 1.3 ⚠️ Cạm bẫy về con số 0.5%

Con số "house edge 0.5%" bạn hay thấy trên mạng là của game **đầy đủ có Split**. Bản này không có Split, nên **−1.087% mới là đúng**. Đừng để ai bảo bạn sai vì họ nhớ con số 0.5%.

Split đáng khoảng +0.6% EV. Cố ý không làm vì state space bùng nổ và lợi ích cận biên thấp.

## 1.4 Bảng chiến thuật

**Hard totals khớp 100%** basic strategy, kể cả các ô tinh tế:
- `11 vs A` là **H** không phải D (đúng cho luật S17; nếu H17 mới là D)
- `10 vs 10` là **H**
- `9` chỉ D vs 3–6
- Ranh giới hàng 12: H vs 2,3 nhưng S vs 4,5,6

## 1.5 Hai ô soft lệch — đã điều tra

`soft 15 vs 4` và `soft 13 vs 5`: bảng published nói **D**, ta nói **H**.

**Đã kiểm bằng tính tay độc lập** — `double_values()` cho `0.125954`, tính tay cũng `0.125954`. **Không phải bug.**

| ô | Q(hit) | Q(double) | chênh |
|---|---|---|---|
| soft 15 vs 4 | 0.05929 | 0.05843 | **8.6 bps** |
| soft 13 vs 5 | 0.13336 | 0.12595 | **74 bps** |

Ô đầu gần như hoà tuyệt đối — bất kỳ khác biệt nhỏ nào về luật hay số bộ bài cũng lật được.

Ô thứ hai thì 74 bps là **quá lớn để giải thích bằng sai số nhỏ**. Giả thuyết: bảng published tính cho 6 bộ hữu hạn, còn ta dùng bộ vô hạn, nên **effect of removal** (bạn đang cầm A và một lá thấp, tức đã rút hai lá thấp khỏi bộ) không được tính. Nhưng ước lượng độ lớn hiệu ứng đó trên shoe 6 bộ chỉ cỡ vài phần nghìn về xác suất — khó tạo ra 74 bps.

**Đây là câu hỏi còn để ngỏ, được ghi rõ trong README.** Nêu chủ động chỗ lệch mạnh hơn nhiều so với để người khác tự tìm ra.

## 1.6 Bối cảnh: bạn thắng bao nhiêu ván?

Chơi 500.000 ván hoàn hảo:

| | |
|---|---|
| **thắng** | 43.3% |
| thua | 48.0% |
| hoà | 8.7% |
| bạn tự quắc | 17.0% |
| **độ lệch chuẩn mỗi ván** | **0.98 = 41× lợi thế** |

Thua là ở **trung bình**, không phải từng ván. Độ lệch chuẩn gấp 41 lần lợi thế nghĩa là ngắn hạn hoàn toàn có thể lãi to. Đó là lý do sòng bài tồn tại: người chơi thắng đủ thường xuyên để quay lại, còn sòng bài chơi hàng triệu ván nên luật số lớn luôn đứng về phía họ.

## 1.7 Lợi thế nhà cái đến từ đúng một luật

Bạn **phải hành động trước**. Đo được: **3.99% số ván là "cả hai cùng quắc"** — hiện tại bạn thua sạch những ván đó.

Nếu luật cho hoà, bạn được lại 3.99%, tức từ **−2.42% thành +1.57%**.

Một luật duy nhất, đáng giá hơn toàn bộ lợi thế nhà cái. Sòng bài không cần chia bài gian — họ chỉ cần bắt bạn đi trước.

## 1.8 Chơi giỏi đáng giá bao nhiêu

| chiến lược | EV |
|---|---|
| **tối ưu** | **−2.42%** |
| bắt chước dealer (rút dưới 17) | −5.68% |
| không bao giờ chịu quắc (dừng từ 12) | −16.04% |

Chơi tối ưu **giảm lỗ hơn một nửa** so với cách hầu hết người ở bàn đang chơi.

---

# PHẦN 2 — Phase 2: Q-Learning

## 2.1 Kết quả (seed 42, 5 triệu ván)

| | |
|---|---|
| mean squared error vs $V^*$ | 6.5 × 10⁻⁵ |
| sai số lớn nhất | 6.29 × 10⁻² |
| khớp chính sách | **98.5%** (197/200) |
| **chi phí sai lệch** | **2.47 bps** |

## 2.2 Nghĩa là gì

Một thuật toán khởi đầu **không biết gì về luật chơi** — không biết xác suất bài, không biết luật dealer — chỉ chơi 5 triệu ván và nhận kết quả cuối ván, đã mò ra gần đúng cùng bảng mà DP tính chính xác.

## 2.3 ⚠️ 98.5% KHÔNG có nghĩa "thua 1.5%"

Đây là lỗi đọc phổ biến nhất.

| đại lượng | giá trị | đo cái gì |
|---|---|---|
| khớp chính sách | 98.5% | 197/200 **ô** ra cùng quyết định |
| MSE | 6.5e−05 | sai số bình phương của **giá trị** |
| **chi phí EV** | **2.47 bps = 0.0247%** | **tiền thực sự mất** |

- Chính sách tối ưu: EV = −2.4208%
- Chính sách RL: EV = −2.4455%
- Chênh: **0.0247%**

Đọc "1.5% ô sai" thành "thua 1.5% tiền" là thổi phồng **60 lần**.

**Bằng tiền:** chơi 10.000 ván cược 1 đơn vị — người hoàn hảo mất 242 đơn vị, agent mất 244.5.

## 2.4 Vì sao chênh lệch nhỏ

Chi phí = **xác suất rơi vào ô đó** × **chênh lệch giá trị**, và cả hai đều nhỏ.

Agent sai **chính xác ở nơi sự thật gần như mơ hồ**. So sánh:

| ô | chênh lệch thật | sai số agent | tỷ lệ tín hiệu/nhiễu |
|---|---|---|---|
| hard 20 vs 6 | 1.55759 | 0.00556 | **280×** → không thể sai |
| hard 12 vs 6 | 0.01683 | 0.01433 | ~1× → **tung đồng xu** |

Ở `hard 20 vs 6` tín hiệu lớn gấp 280 lần nhiễu. Ở `hard 12 vs 6` sai số **lớn ngang** chênh lệch cần phân biệt. Nó không sai vì code hỏng — **nó sai vì thứ cần phân biệt nhỏ hơn nhiễu thống kê của chính nó.**

## 2.5 Chi phí phụ thuộc ô nào sai, không phụ thuộc bao nhiêu ô

Ba seed, 5 triệu ván mỗi cái:

| seed | khớp | chi phí | ô sai |
|---|---|---|---|
| 42 | 98.5% | 2.47 bps | soft 18 v A, soft 18 v 2, hard 12 v 6 |
| 7 | 99.0% | **0.44 bps** | hard 16 v 10, hard 12 v 4 |
| 2024 | 98.5% | 1.90 bps | soft 18 v 5, hard 12 v 4, soft 18 v 4 |

Số ô sai gần như không đổi (2 hoặc 3), còn chi phí dao động **5.6 lần**.

Seed 7 tình cờ sai đúng hai ô **rẻ nhất bảng** — `hard 16 vs 10` (chênh thật 0.0006) và `hard 12 vs 4` (0.0025) — nên gần như miễn phí.

**Đây là lập luận cho việc báo cáo bằng bps:** "ba ô sai" tương thích với bất cứ giá trị nào từ 0.44 đến 2.47 bps, nên đếm ô **không trả lời được câu hỏi người ta thực sự có**.

> ⚠️ Nó **không** phải lập luận rằng bps ổn định hơn. Một phiên bản trước của phân tích này, chạy trước khi sửa định nghĩa α, tưởng như cho thấy đúng điều đó — chi phí bị ghim trong dải hẹp còn số ô nhảy loạn. Pattern đó **không sống sót** qua việc sửa. Cả hai chỉ số đều nhiễu ở mẫu này. Một trong hai có đơn vị nghĩa là gì.

## 2.6 Phát hiện về ω

Với ω = 0.6, sai số chững lại và **mọi ô sai đều sai cùng một hướng** (agent Hit chỗ tối ưu Stand). Lệch một chiều là **bias**, không phải nhiễu → train lâu hơn không cứu được.

**Quét ω tại ngân sách 1 triệu ván:**

| ω | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|
| MSE | 2.0e−3 | 9.0e−4 | 4.2e−4 | **2.2e−4** | 2.3e−4 |

Cải thiện khoảng **9 lần** từ ω = 0.6.

**Chú ý đường cong phẳng lại và hơi đảo ở đỉnh:** ω = 0.9 và 1.0 không phân biệt được (2.2e−4 vs 2.3e−4, khoảng cách 2% mà nhiễu seed che được).

Có lý do: ở ω = 1 **mọi mẫu mang trọng số bằng nhau mãi mãi**, kể cả những mẫu đầu tiên — mà target của chúng được bootstrap từ bảng Q còn rỗng nên sai. ω hơi nhỏ hơn 1 làm phai dần nhiễm bẩn đó. Trung bình Monte Carlo thuần sẽ không có hiệu ứng này, nên đây là hệ quả **riêng của bootstrapping**.

## 2.7 Kiểm chứng định luật $n^{-\omega}$ — cách sai và cách đúng

**Cách SAI (tôi làm đầu tiên):** so ngang qua ω tại cùng $n = 10^6$. Nếu MSE $\propto n^{-\omega}$, tăng ω từ 0.6 lên 1.0 phải chia MSE cho ~400. Thực tế chỉ ~9.

Lý do: $\text{MSE} = C(\omega) \cdot n^{-\omega}$ — hằng số cũng phụ thuộc ω. Độ dốc đo được **−2.49/đơn vị ω**, cách đọc ngây thơ cho **−6.00**.

**Cách ĐÚNG:** theo dõi sai số theo $n$ ở ω cố định. Độ dốc log-log đo được **−1.09** so với **−1.00** lý thuyết. ✅

---

# PHẦN 3 — Phase 3: bộ bài hữu hạn và đếm bài

## 3.1 Edge theo true count (3 triệu ván, chiến lược cố định từ Phase 1)

| true count | ván | edge | khoảng tin cậy 95% |
|---|---|---|---|
| < −3 | 262.112 | −3.429% | [−3.861, −2.998] |
| −3..−2 | 202.822 | −2.211% | [−2.698, −1.724] |
| −2..−1 | 373.115 | −1.890% | [−2.248, −1.533] |
| −1..0 | 567.873 | −0.973% | [−1.262, −0.684] |
| 0..1 | 796.606 | −0.942% | [−1.186, −0.699] |
| 1..2 | 352.521 | −0.404% | [−0.769, −0.039] |
| 2..3 | 193.357 | **+0.267%** | [−0.222, +0.757] |
| ≥ 3 | 251.594 | **+0.890%** | [+0.463, +1.317] |

**Edge tăng đơn điệu qua cả 8 bin và đảo dấu quanh TC ≈ +2.** Đây là bằng chứng thực nghiệm rằng đếm bài hoạt động.

## 3.2 Kiểm chứng chéo quan trọng nhất của dự án

| | |
|---|---|
| edge tổng thể trên shoe hữu hạn | **−1.074%** |
| EV tính chính xác ở Phase 1 | **−1.087%** |
| chênh lệch | **1.3 bps** |

Với 3 triệu ván và σ ≈ 1, sai số chuẩn của trung bình là $1/\sqrt{3\times10^6} \approx 0.058\%$ = 5.8 bps. Chênh 1.3 bps nằm **trong 1 sai số chuẩn**.

**Vì sao điều này mạnh:** hai con số đến từ hai phép tính **hoàn toàn không liên quan** — một bên là đại số trên xác suất cố định, một bên là thống kê từ 3 triệu ván mô phỏng với shoe trôi dạt. Khớp nhau nghĩa là một lỗi phải xuất hiện ở cả hai, theo cùng một cách, độc lập.

## 3.3 Phân phối true count

| bin | < −3 | −3..−2 | −2..−1 | −1..0 | 0..1 | 1..2 | 2..3 | ≥3 |
|---|---|---|---|---|---|---|---|---|
| tần suất | 8.7% | 6.8% | 12.4% | 18.9% | **26.6%** | 11.8% | 6.4% | 8.4% |

Hình chuông quanh 0. Đó là lý do các bin đuôi thiếu dữ liệu.

## 3.4 Index plays học được (8 triệu ván, count trong state)

State space 200 → **1600**. Agent lệch khỏi chính sách bộ-bài-vô-hạn ở **51 ô**. Xếp theo lượng dữ liệu hỗ trợ:

| tay bài | true count | thay đổi | ván đã thấy |
|---|---|---|---|
| **hard 16 vs 10** | **0..1** | hit → **stand** | **68.681** |
| hard 15 vs 10 | 1..2 | hit → stand | 32.127 |
| hard 16 vs 10 | 1..2 | hit → stand | 30.770 |
| hard 13 vs 2 | 0..1 | stand → hit | 20.104 |
| hard 12 vs 6 | −1..0 | stand → hit | 10.948 |

**Dòng đầu là index play nổi tiếng nhất trong blackjack:** stand trên 16 chống lá 10 khi true count đạt 0. Agent tìm ra nó **từ con số 0**, ở đúng ngưỡng mà bảng published ghi.

Chiều của mọi sai lệch đều nhất quán: 14/15/16 chuyển sang Stand khi shoe giàu lá 10 (rút thì dễ quắc), 12/13 chuyển sang Hit khi shoe giàu lá thấp.

## 3.5 ⚠️ Nhưng đuôi thiếu dữ liệu và không nên tin

Ô ít thăm nhất có **115 ván**, trung vị **3.189**. Phase 2 đã xác lập nhiễu của agent ở mức ~0.007 — **lớn hơn nhiều chênh lệch giá trị cần phân giải**.

Sai lệch ở bin trung tâm là thật. Sai lệch ở bin đuôi là **nhiễu đội lốt chính sách**.

Nêu giới hạn này hữu ích hơn là trình bày cả bảng 1600 ô như thể mọi ô đều đáng tin.

---

# PHẦN 4 — Phase 4: Kelly và rủi ro

## 4.1 Gate chặn 7 trong 8 bin

| bin | edge | cận dưới CI | bị chặn? | f (half Kelly) |
|---|---|---|---|---|
| < −3 → 1..2 | âm | âm | ✅ chặn | 0% |
| 2..3 | +0.267% | **−0.222%** | ✅ chặn | 0% |
| ≥ 3 | +0.890% | **+0.463%** | ❌ qua | **0.373%** |

Bin 2..3 có **điểm ước lượng dương** nhưng khoảng tin cậy **chạm 0**, nên không được cược. Đây là gate hoạt động đúng thiết kế.

## 4.2 Phát hiện quan trọng nhất: phải tránh bin âm, không phải cược to bin dương

Phân rã EV ở mức cược đều 1 đơn vị:

| bin | tần suất | edge | đóng góp vào EV |
|---|---|---|---|
| < −3 | 8.7% | −3.429% | −0.2996% |
| −3..−2 | 6.8% | −2.211% | −0.1495% |
| −2..−1 | 12.4% | −1.890% | −0.2351% |
| −1..0 | 18.9% | −0.973% | −0.1841% |
| 0..1 | 26.6% | −0.942% | −0.2502% |
| 1..2 | 11.8% | −0.404% | −0.0475% |
| 2..3 | 6.4% | +0.267% | **+0.0172%** |
| ≥ 3 | 8.4% | +0.890% | **+0.0746%** |
| **tổng** | | | **−1.0742%** |

Hai bin dương góp **+0.092%**. Sáu bin âm lấy **−1.166%**.

**Kết quả mô phỏng bankroll (400 đường × 5000 ván, vốn 1000):**

| chiến lược | vốn cuối (trung vị) | MDD |
|---|---|---|
| buộc cược 1 mọi ván | **957.7** (lỗ 4.2%) | 14.5% |
| ngồi ngoài khi count xấu | **1016.1** (lãi 1.6%) | 9.0% |
| ngồi ngoài + full Kelly | **1025.7** (lãi 2.6%) | 17.3% |

**Biến thiên cược đơn thuần, với cược tối thiểu bắt buộc, KHÔNG thắng được nhà cái dưới bộ luật này. Quyền từ chối mới thắng.**

Full Kelly tăng trưởng nhanh hơn half Kelly nhưng drawdown gần gấp đôi — đúng thứ tự lý thuyết dự đoán, và nếu không thấy pattern này thì đó là dấu hiệu có bug.

## 4.3 Tách đóng góp của cược và của chiến thuật — bảng 2×2

Bốn cấu hình trên **cùng seed**, nên mọi cấu hình thấy cùng bộ bài cùng thứ tự:

| vốn cuối (trung bình) | cược đều | cược Kelly |
|---|---|---|
| chiến thuật cố định | 956.4 (cơ sở) | 1017.9 |
| chiến thuật theo count | 960.8 | 1021.0 |

Vì chung seed, ta so **theo cặp** (paired) — loại bỏ phần lớn phương sai giữa các đường:

| hiệu ứng | chênh lệch paired | sai số chuẩn | t |
|---|---|---|---|
| **định cỡ cược** | **+61.45** | 4.80 | **+12.80** ✅ |
| chiến thuật theo count | +4.41 | 2.53 | +1.74 ❌ |

Định cỡ cược **rõ ràng**. Chiến thuật theo count là hiệu ứng dương nhỏ mà **mẫu này không phân giải được** — t = 1.74 không vượt ngưỡng, và cần khoảng 4 lần số đường mới tách được khỏi 0.

### Hai ghi chú phương pháp, cả hai đều là lỗi tôi mắc rồi sửa

**Dùng trung bình paired, không dùng trung vị.** So trung vị của bốn phân phối độc lập cho hiệu ứng chiến thuật là **−2.0** — **sai dấu**. Trung vị là thống kê tốt để mô tả *một* phân phối nhưng là estimator nhiễu cho *hiệu* giữa hai.

**Không tính phần trăm từ một đại lượng không có ý nghĩa thống kê.** Phiên bản trước báo "97% từ cược, 3% từ chiến thuật" và so với 90/10 của Griffin/Wong. Con số đó **không đứng được**: không thể lấy tỷ lệ của một đại lượng không phân biệt được với 0, và mẫu số dựng từ ước lượng còn chưa chắc dấu. Phát biểu trung thực là **chiều và mức ý nghĩa**, để trống tỷ lệ tới khi có đủ dữ liệu.

## 4.4 Kelly fraction — số thật

| | |
|---|---|
| μ (bin ≥3) | 0.00890 |
| E[X²] | 1.19474 |
| Taylor: μ/E[X²] | 0.00745 |
| **nghiệm số chính xác** | **0.00746** |
| sai lệch | under-bet 0.1% |
| E[X³] (moment bậc 3) | **+0.213** (xiên dương) |

Half Kelly → **0.373% bankroll**, tức 3.73 đơn vị trên vốn 1000, so với cược tối thiểu 1 đơn vị.

## 4.5 Rủi ro

| | |
|---|---|
| VaR 99% | 144 |
| **CVaR 99%** | **173** |
| khoảng bootstrap 95% cho CVaR | **[143, 188]** |

Khoảng bootstrap được báo cáo vì phân vị 99% từ 400 đường dựa trên khoảng **4 quan sát**. Điểm ước lượng một mình sẽ thổi phồng độ chính xác.

## 4.6 Chỗ lý thuyết KHÔNG áp dụng — và tại sao

Công thức đóng dự đoán ruin **12.5%** (half Kelly) và **50%** (full Kelly). Đo thực tế: **0% ở mọi cấu hình.**

**Không phải bug** — xấp xỉ đơn giản là không áp dụng được. Nó giả định thời gian liên tục, cược chia nhỏ vô hạn, edge biết chính xác, và **horizon vô hạn**.

Ở đây horizon là 5.000 ván, chỉ ~8% được cược trên mức tối thiểu, và Kelly chỉ đòi dưới 1% bankroll. Tổng exposure quá nhỏ để bankroll tiến gần một nửa giá trị ban đầu — chạm ngưỡng sẽ là biến cố cỡ **3 sigma**.

Báo cáo khoảng cách và nguyên nhân hữu ích hơn là im lặng bỏ qua phép so sánh.

---

# PHẦN 5 — Ba đường kiểm chứng độc lập

Con số EV là thứ có thể sai theo cách vẫn trông hợp lý, nên nó được kiểm ba cách:

**1. So với số đã công bố.** −2.421% và −1.087% cho bộ luật này. Khớp 5 chữ số.

**2. So với mô phỏng.** Solver tính xác suất bằng tay và không mô phỏng; môi trường mô phỏng và không dùng xác suất đó. Chơi chính sách tối ưu 300.000 ván rơi trong **3 sai số chuẩn** của giá trị chính xác. Một lỗi phải xuất hiện ở cả hai, cùng cách, độc lập.

**3. So với phương pháp model-free.** Q-Learning khởi đầu từ không gì ngoài phần thưởng lấy mẫu, và tới cùng hàm giá trị.

**Cộng thêm ở Phase 3:** edge tổng thể đo trên shoe hữu hạn (−1.074%) khớp EV tính chính xác (−1.087%) trong 1 sai số chuẩn.

---

# PHẦN 6 — Bảng tra nhanh mọi con số

| | |
|---|---|
| P(lá 10) | 4/13 |
| số trạng thái quyết định (vô hạn) | 200 |
| số trạng thái (shoe hữu hạn, đầy đủ) | 7.4 × 10¹⁶ |
| VI hội tụ | 13 sweeps |
| EV Hit/Stand | **−2.421%** |
| EV có Double | **−1.087%** |
| giá trị quyền Double | +1.334% |
| thắng / thua / hoà | 43.3% / 48.0% / 8.7% |
| tự quắc | 17.0% |
| σ mỗi ván | 0.98 |
| "cả hai cùng quắc" | 3.99% |
| bắt chước dealer | −5.68% |
| không chịu quắc | −16.04% |
| dealer quắc, ngửa 6 | 42.3% |
| dealer quắc, ngửa 9 | 22.8% |
| V\*(20, dealer 6) | +0.704 |
| V\*(16, dealer 10) | −0.540 |
| MSE Q-Learning (seed 42, 5M) | 6.47e−05 |
| khớp chính sách | 98.5% |
| **chi phí sai lệch** | **2.47 bps** |
| house edge để so sánh | 242 bps |
| độ dốc log-log (ω=1) | −1.09 (lý thuyết −1.00) |
| edge TC < −3 | −3.429% |
| edge TC ≥ +3 | **+0.890%** |
| edge đảo dấu tại | TC ≈ +2 |
| edge tổng thể shoe hữu hạn | −1.074% |
| ván cần/bin để đo edge 1% | 38.416 |
| index play `16 v 10` | Stand khi TC ≥ 0 |
| Kelly f\* (bin ≥3, half) | 0.373% bankroll |
| E[X³] (bin ≥3) | +0.213 |
| VaR 99% | 144 |
| CVaR 99% | 173, CI [143, 188] |
| hiệu ứng định cỡ cược (paired) | +61.45, t = 12.80 |
| hiệu ứng chiến thuật theo count | +4.41, t = 1.74 (không ý nghĩa) |
| số test | 117 |

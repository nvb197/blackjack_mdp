# Toán học của dự án — từ gốc, đầy đủ

Tài liệu này giả định bạn biết đại số tuyến tính cơ bản, xác suất cơ bản (kỳ vọng, phương sai, phân phối), và giải tích cơ bản (đạo hàm). Mọi khái niệm khác — kể cả những khái niệm "hiển nhiên" với dân RL hay quant — được giải thích từ đầu.

Đọc cùng lúc với `02_HIEU_CODE.md` (nơi có code) và `07_THUAT_NGU_VA_KHAI_NIEM.md` (nơi có định nghĩa nhanh nếu bạn quên một từ giữa chừng).

---

# MỤC LỤC

- Phần 1: Markov Decision Process — ngôn ngữ chung của cả dự án
- Phần 2: Phương trình Bellman — trái tim của lý thuyết
- Phần 3: Value Iteration — thuật toán giải chính xác
- Phần 4: Q-Learning — học khi không biết luật
- Phần 5: Lý thuyết hội tụ của Q-Learning — vì sao nó *phải* đúng
- Phần 6: Đếm bài — biến một bài toán dừng thành không dừng
- Phần 7: Suy luận thống kê — vì sao 3 triệu ván mới đủ
- Phần 8: Kelly Criterion — cược bao nhiêu
- Phần 9: Đo lường rủi ro — VaR, CVaR, và các tiên đề
- Phần 10: Ba chủ đề nối tất cả lại với nhau

---

# PHẦN 1 — Markov Decision Process (MDP)

## 1.1 Vấn đề tổng quát mà MDP mô tả

Rất nhiều bài toán quan trọng có chung một hình dạng: bạn **quan sát** một tình huống, **hành động**, tình huống **thay đổi một cách ngẫu nhiên** (một phần do hành động của bạn, một phần do may rủi), rồi bạn quan sát tình huống mới và lại hành động. Cứ thế cho tới khi kết thúc, và bạn nhận được **phần thưởng** dọc đường hoặc ở cuối.

Ví dụ ngoài blackjack: một nhà giao dịch quan sát giá và vị thế hiện tại, quyết định mua/bán/giữ, thị trường phản ứng một phần theo lệnh của họ và một phần ngẫu nhiên, họ nhận lãi/lỗ. Một robot quan sát vị trí, chọn hướng di chuyển, môi trường (có thể có ma sát, gió) đẩy nó tới vị trí mới.

MDP là khung toán học **chuẩn hoá** hình dạng chung này để ta có công cụ giải nó.

## 1.2 Bốn thành phần chính thức

Một MDP được xác định bởi bộ bốn $(\mathcal{S}, \mathcal{A}, P, R)$:

| Ký hiệu | Tên | Ý nghĩa | Trong blackjack |
|---|---|---|---|
| $\mathcal{S}$ | tập trạng thái | mọi tình huống có thể | (tổng điểm, lá ngửa dealer, có ace=11 hay không) |
| $\mathcal{A}$ | tập hành động | mọi lựa chọn có thể | {Stand, Hit, Double} |
| $P(s' \mid s, a)$ | hàm chuyển | xác suất tới $s'$ nếu ở $s$ làm $a$ | tính từ xác suất rút bài |
| $R(s, a, s')$ | hàm thưởng | phần thưởng nhận được | +1/-1/0/+1.5/±2 tuỳ kết quả ván |

Đôi khi thêm thành phần thứ năm, **hệ số chiết khấu** $\gamma \in [0,1]$, quyết định phần thưởng tương lai đáng giá bao nhiêu so với phần thưởng ngay bây giờ. Trong dự án này $\gamma = 1$ (không chiết khấu) vì ván bài luôn kết thúc trong hữu hạn bước — không cần "làm nhẹ" phần thưởng tương lai để tổng không phát tán ra vô cùng.

## 1.3 Tính chất Markov — định nghĩa chính xác

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)$$

Nói bằng lời: **tương lai chỉ phụ thuộc trạng thái hiện tại, không phụ thuộc cách bạn tới được đó.**

Đây không phải giả định "hiển nhiên đúng" — nó là một **lựa chọn thiết kế trạng thái**. Với bất kỳ quá trình ngẫu nhiên nào, bạn luôn có thể *cưỡng bức* tính Markov bằng cách nhét toàn bộ lịch sử vào trạng thái (trạng thái = "mọi thứ đã xảy ra từ đầu tới giờ"). Điều đó luôn đúng về mặt toán học nhưng vô dụng, vì không gian trạng thái sẽ bùng nổ.

**Nghệ thuật thật sự là tìm một trạng thái ĐỦ NHỎ mà vẫn giữ tính Markov.**

## 1.4 Vì sao 3 con số là đủ (bộ bài vô hạn)

Với bộ bài vô hạn (rút có hoàn lại), xác suất rút mỗi lá bài **không bao giờ thay đổi**, bất kể bao nhiêu lá đã rút trước đó. Nên nếu bạn biết:

1. Tổng điểm hiện tại của bạn
2. Lá ngửa của dealer
3. Bạn có đang giữ một ace tính là 11 hay không

...thì bạn biết **chính xác** phân phối xác suất của mọi thứ xảy ra tiếp theo. Việc bạn có A+5 hay 2+2+2+... không quan trọng — chỉ tổng điểm và cấu hình soft/hard quyết định tương lai.

Ta gọi bộ ba này là một **thống kê đủ** (sufficient statistic): nó tóm tắt toàn bộ thông tin liên quan từ lịch sử, vứt bỏ phần không liên quan (thứ tự các lá cụ thể), mà vẫn giữ đủ để dự đoán tương lai chính xác như khi biết toàn bộ lịch sử.

**Số trạng thái quyết định:** tổng từ 12–21 (10 giá trị) × lá ngửa từ A–10 (10 giá trị) × soft/hard (2 giá trị) = **200**.

Vì sao chỉ tính từ 12? Vì với tổng dưới 12 (và không có ace tính 11), rút thêm **không bao giờ khiến bạn quắc** — ngay cả rút lá 10 cũng chỉ đưa bạn lên tối đa 21. Hit ở đây trội tuyệt đối so với Stand (Stand chỉ có thể hoà hoặc thua nếu dealer đạt ≥12, còn Hit chỉ có thể tốt hơn hoặc bằng). Không có gì phải cân nhắc, nên không cần lưu.

## 1.5 Với bộ bài hữu hạn — tại sao 3 con số không còn đủ

Rút một lá **không hoàn lại** làm thay đổi xác suất rút lá tiếp theo. Cụ thể: nếu vừa rút một lá 6, thì xác suất lá tiếp theo cũng là 6 giảm xuống một chút, vì giờ chỉ còn 23 lá 6 trong 311 lá còn lại thay vì 24 trong 312.

Nên để giữ tính Markov, trạng thái phải bao gồm **toàn bộ thành phần shoe còn lại** — biết chính xác còn bao nhiêu lá mỗi hạng.

**Đếm số trạng thái khả dĩ của thành phần shoe:** mỗi hạng 1–9 có thể còn từ 0 đến 24 lá (25 khả năng), hạng 10 có thể còn từ 0 đến 96 lá (97 khả năng — nhưng thực ra bị ràng buộc bởi tổng số lá đã rút, nên con số thật nhỏ hơn một chút, tuy vẫn cùng bậc độ lớn):

$$25^9 \times 97 \approx 3.7 \times 10^{14}$$

Nhân với 200 trạng thái người chơi:

$$3.7 \times 10^{14} \times 200 \approx 7.4 \times 10^{16}$$

Lưu mỗi trạng thái ở 8 byte (một số thực chuẩn double): khoảng **590 petabyte** — nhiều hơn dung lượng lưu trữ của hầu hết trung tâm dữ liệu lớn trên thế giới, cho **một bài toán đồ chơi**.

**Kết luận quan trọng cần phát biểu đúng:** bộ bài hữu hạn **không phá vỡ** tính Markov về mặt lý thuyết — quá trình vẫn Markov nếu bạn định nghĩa trạng thái đủ lớn. Vấn đề là trạng thái đủ để giữ Markov **bùng nổ về số chiều**, khiến việc giải chính xác trở thành bất khả thi về mặt tính toán, dù khả thi về mặt lý thuyết.

Đây là một câu hỏi bẫy kinh điển: nếu bạn trả lời "bộ bài hữu hạn phá vỡ Markov", bạn sai. Câu trả lời đúng dài hơn nhưng chính xác hơn nhiều.

## 1.6 Hàm chính sách (policy)

Một **chính sách** $\pi$ là một quy tắc ánh xạ từ trạng thái sang hành động (hoặc, tổng quát hơn, sang một phân phối xác suất trên các hành động). Chính sách **tất định** (deterministic) chọn đúng một hành động cho mỗi trạng thái: $\pi(s) = a$. Chính sách **ngẫu nhiên** (stochastic) cho một phân phối: $\pi(a \mid s) = P(\text{chọn } a \mid s)$.

Trong dự án này, chính sách tối ưu $\pi^*$ mà Value Iteration tìm ra là tất định — luôn có một hành động tốt nhất rõ ràng (trừ khi hai hành động có giá trị bằng nhau tuyệt đối, trường hợp đo được là hiếm).

---

# PHẦN 2 — Phương trình Bellman

## 2.1 Định nghĩa hàm giá trị

$$V^\pi(s) = \mathbb{E}\Big[\sum_{t=0}^{T} \gamma^t r_t \,\Big|\, s_0 = s, \text{theo chính sách } \pi\Big]$$

Nói bằng lời: **giá trị của trạng thái $s$ dưới chính sách $\pi$ là tổng phần thưởng kỳ vọng, tính từ $s$ trở đi, nếu bạn luôn hành động theo $\pi$.**

Trong blackjack, $T$ luôn hữu hạn (ván kết thúc) và $\gamma = 1$, nên đơn giản còn:

$$V^\pi(s) = \mathbb{E}[\text{tổng phần thưởng từ } s \text{ tới hết ván} \mid \pi]$$

**Hàm giá trị tối ưu:**

$$V^*(s) = \max_\pi V^\pi(s)$$

Giá trị tốt nhất có thể đạt được từ $s$, trên mọi chính sách khả dĩ.

## 2.2 Hàm giá trị hành động (Q-function)

$$Q^\pi(s, a) = \mathbb{E}\Big[r_0 + \gamma V^\pi(s_1) \,\Big|\, s_0 = s, a_0 = a\Big]$$

Khác biệt tinh tế với $V$: $Q(s,a)$ là giá trị nếu bạn làm hành động $a$ **cụ thể** ở $s$ (không nhất thiết theo chính sách), rồi từ đó trở đi theo chính sách $\pi$.

Quan hệ giữa $V$ và $Q$:

$$V^\pi(s) = Q^\pi(s, \pi(s)) \qquad V^*(s) = \max_a Q^*(s,a)$$

Vì sao cần cả hai? $V$ cho biết trạng thái tốt tới đâu. $Q$ cho biết **hành động** tốt tới đâu — và đó chính là thứ bạn cần để **ra quyết định**. Nếu chỉ có $V$, bạn vẫn phải biết $P(s'|s,a)$ để tính hành động nào tốt nhất. Nếu có $Q$ luôn, bạn chỉ cần lấy $\arg\max_a Q(s,a)$ — không cần biết gì về xác suất chuyển nữa. Đây là lý do Q-Learning (Phần 4) học $Q$ chứ không học $V$: nó không có quyền truy cập $P(s'|s,a)$.

## 2.3 Phương trình Bellman kỳ vọng

Với một chính sách $\pi$ cố định, giá trị của $s$ có thể viết đệ quy theo giá trị của các trạng thái kế tiếp:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \big[r(s,a,s') + \gamma V^\pi(s')\big]$$

Đây gọi là phương trình Bellman **kỳ vọng** (expectation) vì nó lấy trung bình có trọng số trên cả hành động (theo $\pi$) và trạng thái kế tiếp (theo $P$).

## 2.4 Phương trình Bellman tối ưu

Thay vì lấy trung bình trên hành động theo $\pi$, ta **chọn hành động tốt nhất**:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \big[r(s,a,s') + \gamma V^*(s')\big]$$

Đây là phương trình trung tâm của toàn bộ lý thuyết MDP. Nó nói: **giá trị tối ưu của một trạng thái bằng giá trị của hành động tốt nhất, mà giá trị của một hành động lại phụ thuộc vào giá trị tối ưu của các trạng thái kế tiếp.**

Đây là định nghĩa **đệ quy** (đệ quy nghĩa là định nghĩa một thứ bằng chính nó, ở quy mô nhỏ hơn) — và đó chính là vấn đề con gà–quả trứng đã nhắc ở Phần 1.

## 2.5 Toán tử Bellman và tính chất co (contraction)

Định nghĩa **toán tử Bellman** $T$: với bất kỳ hàm $V$ nào (không nhất thiết tối ưu), $T$ biến nó thành một hàm mới:

$$(TV)(s) = \max_a \sum_{s'} P(s'|s,a)\big[r(s,a,s') + \gamma V(s')\big]$$

$V^*$ chính là **điểm bất động** (fixed point) của $T$: $TV^* = V^*$.

**Định lý ánh xạ co Banach** (Banach fixed-point theorem): nếu $T$ là một **ánh xạ co** trên một không gian metric đầy đủ — nghĩa là tồn tại hằng số $\gamma < 1$ sao cho

$$\|TV_1 - TV_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty \quad \text{với mọi } V_1, V_2$$

— thì $T$ có **đúng một** điểm bất động, và lặp $V_{k+1} = TV_k$ từ **bất kỳ** điểm khởi đầu nào cũng hội tụ về điểm bất động đó, với tốc độ hình học.

**Tại sao toán tử Bellman là ánh xạ co khi $\gamma < 1$:** trực giác là mỗi lần áp $T$, sai số giữa hai ước lượng bị "chiết khấu" đi một hệ số $\gamma$ trước khi lan sang trạng thái kế tiếp. Sau vô hạn lần lặp, sai số ban đầu bị nhân với $\gamma^k \to 0$.

**Chứng minh nhanh (cho ai muốn thấy):** Với hai hàm $V_1, V_2$ bất kỳ,

$$|(TV_1)(s) - (TV_2)(s)| = \Big|\max_a \sum_{s'} P(s'|s,a)[r + \gamma V_1(s')] - \max_a \sum_{s'} P(s'|s,a)[r + \gamma V_2(s')]\Big|$$

Dùng bất đẳng thức $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$ (hàm max không làm khoảng cách lớn hơn khoảng cách lớn nhất giữa các thành phần), ta có

$$\le \max_a \Big|\sum_{s'} P(s'|s,a) \gamma [V_1(s') - V_2(s')]\Big| \le \gamma \max_a \sum_{s'} P(s'|s,a) \|V_1-V_2\|_\infty = \gamma \|V_1-V_2\|_\infty$$

(vì $\sum_{s'} P(s'|s,a) = 1$). Đây đúng là định nghĩa của ánh xạ co với hằng số $\gamma$.

## 2.6 Câu hỏi bẫy: $\gamma = 1$ thì mất tính co, vậy sao vẫn hội tụ?

Với $\gamma = 1$, chứng minh trên chỉ cho $\le 1 \cdot \|V_1-V_2\|_\infty$ — **không co**, chỉ **không nở ra**. Định lý Banach không áp dụng được trực tiếp, vì nó cần $\gamma$ **thực sự nhỏ hơn** 1.

**Nhưng blackjack (và nhiều bài toán episodic khác) vẫn hội tụ, vì một lý do khác.**

Đây là bài toán thuộc lớp **Stochastic Shortest Path (SSP)**: một quá trình episodic (có trạng thái kết thúc) mà **mọi chính sách đều "proper"** — nghĩa là dưới bất kỳ chính sách nào, xác suất đạt trạng thái kết thúc trong hữu hạn bước là 1 (không có cách nào "chơi mãi mãi" mà không bao giờ kết thúc).

Trong blackjack: mỗi lần Hit, tổng điểm của bạn **tăng nghiêm ngặt** (bạn cộng thêm ít nhất 1 điểm). Tổng điểm bị chặn trên bởi 21 (quá đó là quắc = kết thúc). Nên số lần Hit tối đa trước khi buộc phải kết thúc (hoặc do Stand, hoặc do quắc) là **hữu hạn và bị chặn** (không quá khoảng 20 lần, vì mỗi lá tối thiểu 1 điểm). Không có cách nào lặp vô hạn.

**Lý thuyết SSP** (xem Bertsekas & Tsitsiklis, hoặc Puterman §6.3) chứng minh: nếu mọi chính sách đều proper, và phần thưởng bị chặn, thì toán tử Bellman vẫn có **điểm bất động duy nhất**, dù $\gamma = 1$ và toán tử không co theo nghĩa chuẩn. Chứng minh dùng một chuẩn khác (có trọng số theo "thời gian kỳ vọng tới kết thúc" từ mỗi trạng thái) mà theo chuẩn đó, toán tử **là** co.

**Ghi nhớ gọn:** $\gamma = 1$ mất tính co theo nghĩa Banach chuẩn, nhưng bài toán episodic-chắc-chắn-kết-thúc (SSP với mọi chính sách proper) vẫn có nghiệm duy nhất và Value Iteration vẫn hội tụ, nhờ một lập luận riêng dựa trên tính hữu hạn của thời gian tới kết thúc, không dựa trên ánh xạ co cổ điển.

## 2.7 Định lý cải thiện chính sách (Policy Improvement Theorem)

Một kết quả nền tảng khác, không dùng trực tiếp trong code nhưng đáng biết vì nó giải thích *tại sao* thuật toán "chọn hành động tốt nhất theo ước lượng hiện tại" hội tụ về tối ưu chứ không mắc kẹt ở đâu đó tồi hơn.

**Phát biểu:** nếu $\pi'$ là chính sách "tham lam" theo $Q^\pi$ (nghĩa là $\pi'(s) = \arg\max_a Q^\pi(s,a)$ với mọi $s$), thì $V^{\pi'}(s) \ge V^\pi(s)$ với **mọi** $s$ — chính sách mới không tệ hơn chính sách cũ ở bất kỳ đâu.

Đây là lý do việc lặp đi lặp lại "ước lượng giá trị, rồi hành động tham lam theo ước lượng đó" không bao giờ đi lùi. Value Iteration và Q-Learning đều dựa trên nguyên lý này, dù cách chúng ước lượng giá trị khác nhau hoàn toàn.

---

# PHẦN 3 — Value Iteration

## 3.1 Thuật toán, viết đầy đủ

```
khởi tạo V(s) = 0 với mọi s
lặp:
    với mọi s:
        V_mới(s) = max_a  Σ_s' P(s'|s,a) [r(s,a,s') + γ V(s')]
    nếu max_s |V_mới(s) - V(s)| < θ:  dừng
    V ← V_mới
```

Mỗi lần duyệt toàn bộ trạng thái gọi là một **sweep**.

## 3.2 Vì sao đây là "quy hoạch động" (dynamic programming)

Quy hoạch động là kỹ thuật giải bài toán bằng cách chia thành các bài toán con **chồng lấp** (overlapping subproblems) và lưu lại kết quả bài toán con để không tính lại. "Giá trị của trạng thái $s$" phụ thuộc vào "giá trị của các trạng thái kế tiếp $s'$" — nhưng nhiều trạng thái $s$ khác nhau có thể cùng dẫn tới cùng một $s'$, nên tính giá trị của $s'$ một lần và tái sử dụng là bản chất của DP.

Value Iteration là DP vì ở mỗi sweep, ta dùng $V(s')$ đã tính (dù chưa hội tụ) để tính $V(s)$ mới — không tính lại từ đầu.

## 3.3 Kết quả đo được: hội tụ trong 13 sweeps

| sweep | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| $\|V_{k+1}-V_k\|_\infty$ | 9.6e−1 | 4.2e−1 | 1.6e−1 | 6.1e−2 | 2.1e−2 | 6.2e−3 | 1.2e−3 |

| sweep | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|
| $\|V_{k+1}-V_k\|_\infty$ | 1.5e−4 | 1.5e−5 | 1.4e−6 | 9.8e−8 | 5.3e−9 | 2.1e−10 |

**Vì sao co lại nhanh, và càng về sau càng nhanh hơn:** sweep thứ $k$ về bản chất "lan truyền" thông tin từ các trạng thái cách $k$ bước tới kết thúc. Để một ván còn kéo dài quá $k$ bước, người chơi phải rút liên tiếp $k$ lá đều nhỏ (không quắc) — một chuỗi sự kiện có xác suất giảm theo cấp số nhân theo $k$. Nên "phần đóng góp mới" ở mỗi sweep càng về sau càng nhỏ, và nhỏ theo tốc độ nhanh hơn tuyến tính.

**Kiểm chứng bằng cách đổi ngưỡng dừng:**

| $\theta$ | $10^{-6}$ | $10^{-9}$ | $10^{-12}$ |
|---|---|---|---|
| số sweep | 11 | 13 | 15 |

Mỗi sweep thêm ~1 chữ số thập phân độ chính xác — khớp với việc sai số co theo cấp số nhân đã thấy ở bảng trên.

## 3.4 Vectorization: vì sao code không có vòng lặp Python

Về mặt toán, Value Iteration là một phép lặp trên **440 trạng thái** (22 tổng điểm × 10 lá ngửa × 2 soft/hard, dù chỉ 200 có ý nghĩa quyết định). Viết vòng lặp Python duyệt qua 440 trạng thái, với mỗi trạng thái lại duyệt qua tối đa 10 lá bài có thể rút — sẽ đúng về mặt toán học nhưng **chậm**, vì Python thực thi từng dòng bytecode một cách tuần tự với overhead lớn.

**Vectorization** là kỹ thuật viết lại phép tính đó dưới dạng phép toán trên **toàn bộ mảng cùng lúc**, để NumPy giao việc tính toán thực sự cho code C đã biên dịch sẵn (nhanh hơn Python thuần hàng chục đến hàng trăm lần).

Ví dụ cụ thể trong `dp.py`, hàm `stand_values()`:

```python
sign = np.sign(totals - dealer_finals)      # ma trận 22×5, không vòng lặp
return sign @ dealer[:5, :] + dealer[BUST_IDX, :][None, :]
```

`sign` là ma trận: mỗi phần tử `sign[t, k]` cho biết bạn thắng/hoà/thua nếu tổng của bạn là `t` và dealer dừng ở `k`. Phép `@` là **nhân ma trận** — nó tính tổng có trọng số trên mọi kết cục dealer, cho **mọi** tổng điểm người chơi và **mọi** lá ngửa **cùng một lúc**, bằng một lệnh.

## 3.5 Điều kiện peek — chi tiết toán học

Khi dealer ngửa A hoặc 10, họ kiểm tra lá úp trước khi bạn hành động. Nếu họ có blackjack, ván kết thúc ngay — bạn không bao giờ được ra quyết định trong tình huống đó.

Vậy khi tính phân phối kết cục dealer **để dùng cho quyết định của người chơi**, ta phải tính **xác suất có điều kiện** trên sự kiện "dealer không có blackjack":

$$P(\text{hole}=h \mid \text{không blackjack}) = \frac{P(\text{hole}=h)}{P(\text{không blackjack})} = \frac{P(h)}{1 - P(\text{lá tạo blackjack})}$$

Đây là ứng dụng trực tiếp của **định lý Bayes / xác suất có điều kiện**: $P(A \mid B) = P(A \cap B) / P(B)$, ở đây $A$ = "lá úp là $h$", $B$ = "không có blackjack", và với $h$ không tạo blackjack thì $A \cap B = A$.

**Sai lầm phổ biến:** tính phân phối kết cục dealer *không* điều kiện hoá (dùng $P(h)$ thô), rồi *sau đó* mới loại các trường hợp blackjack ra khỏi phép tính EV tổng. Nghe có vẻ tương đương nhưng **không phải** — vì phân phối kết cục dealer *có điều kiện* dùng để định giá **mọi hành động Hit/Stand** của người chơi ở mọi trạng thái, không chỉ ở bước tính EV cuối. Bỏ sót điều kiện hoá này làm lệch giá trị của *mọi* trạng thái có dealer ngửa A hoặc 10, không chỉ EV tổng thể.

---

# PHẦN 4 — Q-Learning

## 4.1 Vấn đề Value Iteration không giải quyết được

Value Iteration cần biết chính xác $P(s'|s,a)$ — hàm chuyển. Với blackjack bộ bài vô hạn, ta tính được nó bằng tay (Phần 3). Nhưng với hầu hết bài toán thực tế (thị trường tài chính, robot, game phức tạp), **không ai biết** hàm chuyển chính xác.

Q-Learning giải quyết vấn đề đó: nó học $Q(s,a)$ chỉ từ **kinh nghiệm** — chơi thử, quan sát phần thưởng — mà không cần biết $P(s'|s,a)$ dưới bất kỳ hình thức nào.

## 4.2 Temporal-Difference (TD) Learning — ý tưởng cốt lõi

Có hai cách "học" giá trị từ kinh nghiệm:

**Monte Carlo:** chơi hết một ván trọn vẹn, rồi cập nhật giá trị của **mọi** trạng thái đã ghé qua bằng **kết quả cuối cùng thực tế** của ván đó. Đơn giản, không thiên lệch (unbiased), nhưng phải đợi hết ván mới cập nhật được gì, và phương sai cao (kết quả một ván đơn lẻ rất nhiễu).

**Temporal-Difference (TD):** cập nhật **ngay sau mỗi bước**, dùng ước lượng hiện tại của bước kế tiếp làm "gần đúng" cho phần còn lại của ván, thay vì đợi kết quả thật. Đây gọi là **bootstrapping** — dùng một ước lượng để cập nhật một ước lượng khác (cùng loại từ "tự kéo dây giày mình lên", ý nói tự cải thiện dựa trên chính mình).

## 4.3 Công thức cập nhật TD, giải thích từng phần

$$Q(s,a) \leftarrow Q(s,a) + \alpha \underbrace{\Big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{"target" — ước lượng mới}} - \underbrace{Q(s,a)}_{\text{ước lượng cũ}}\Big]}_{\text{TD error } \delta}$$

- **Target** = phần thưởng thật $r$ vừa nhận, cộng với ước lượng giá trị tốt nhất có thể từ trạng thái kế tiếp (chiết khấu). Đây là "phiên bản mới hơn, một bước gần sự thật hơn" của $Q(s,a)$.
- **TD error** $\delta$ = chênh lệch giữa target và ước lượng cũ — "tôi đã sai bao nhiêu".
- **$\alpha$** (learning rate / step size) = đi bao nhiêu phần của quãng đường về phía target.

Đây chính là một dạng **gradient descent ngẫu nhiên** (stochastic gradient descent) trên sai số bình phương $\frac{1}{2}\delta^2$, nếu bạn muốn liên hệ với optimization cổ điển.

## 4.4 Ví dụ số cụ thể — tính tay từng bước

Giả sử tại một ô $(s,a)$, ta quan sát lần lượt 8 phần thưởng: $1, -1, 1, 1, -1, 0, 1.5, -1$ (mỗi lần đều là trạng thái kết thúc, nên target chỉ là $r$, không cộng gì thêm). Dùng $\alpha_n = 1/n$ (với $n$ là số lần thăm, bắt đầu từ 1):

| lần thăm $n$ | $\alpha_n = 1/n$ | phần thưởng $r$ | TD error $\delta$ | $Q$ sau cập nhật |
|---|---|---|---|---|
| 1 | 1.000 | +1.0 | +1.000 | +1.0000 |
| 2 | 0.500 | −1.0 | −2.000 | +0.0000 |
| 3 | 0.333 | +1.0 | +1.000 | +0.3333 |
| 4 | 0.250 | +1.0 | +0.667 | +0.5000 |
| 5 | 0.200 | −1.0 | −1.500 | +0.2000 |
| 6 | 0.167 | 0.0 | −0.200 | +0.1667 |
| 7 | 0.143 | +1.5 | +1.333 | +0.3571 |
| 8 | 0.125 | −1.0 | −1.357 | +0.1875 |

**Q cuối = 0.1875, và trung bình cộng thật của 8 phần thưởng cũng là 0.1875.** *(Bảng này đã kiểm chứng bằng code thật — khớp chính xác test `test_running_estimate_is_the_exact_sample_mean`.)*

Đây chính xác là ý nghĩa của $\alpha_n = 1/n$: nó khiến $Q$ luôn bằng **trung bình cộng của mọi phần thưởng đã quan sát**, cập nhật dần thay vì tính lại từ đầu mỗi lần.

## 4.5 Vì sao trạng thái cuối không được bootstrap

```python
target = r if done else r + np.max(self.Q[nt, nup, nsoft])
```

Nếu $s$ là trạng thái kết thúc, không có "trạng thái kế tiếp" nào để lấy giá trị — ván đã xong. Nên target chỉ là $r$. Nếu vô tình vẫn cộng thêm $\max_{a'} Q(s', a')$ với $s'$ nào đó (ví dụ, tái sử dụng trạng thái cũ do lỗi lập trình), giá trị sai đó sẽ lan ngược qua bootstrapping vào **gần như mọi trạng thái**, vì hầu hết trạng thái trong blackjack chỉ cách kết thúc 1–3 bước.

## 4.6 Off-policy vs on-policy — khác biệt cốt lõi

Có hai khái niệm dễ nhầm:

- **Behaviour policy:** chính sách agent **thực sự dùng** để chọn hành động khi thu thập kinh nghiệm (ở đây là ε-greedy — thường chọn tốt nhất theo ước lượng hiện tại, nhưng đôi khi chọn ngẫu nhiên để khám phá).
- **Target policy:** chính sách mà agent đang **học giá trị của nó**.

**Q-Learning là off-policy:** vì chữ $\max_{a'}$ trong công thức cập nhật, target luôn được tính theo hành động **tốt nhất** ở $s'$ — bất kể agent thực sự chọn hành động nào ở đó khi thu thập dữ liệu. Nghĩa là target policy = chính sách tham lam tuyệt đối, còn behaviour policy = ε-greedy (có khám phá ngẫu nhiên). Hai cái khác nhau, và đó là ý nghĩa của "off-policy" — học về một chính sách trong khi hành xử theo một chính sách khác.

**SARSA là on-policy:** nó thay $\max_{a'} Q(s',a')$ bằng $Q(s', a'_{\text{thực sự chọn}})$ — dùng đúng hành động mà behaviour policy chọn. Nên target policy = behaviour policy = cùng một chính sách ε-greedy.

**Hệ quả thực tế khác biệt hai thuật toán:**

Vì Q-Learning học giá trị của chính sách tham lam bất kể nó hành xử ra sao, nó **không bận tâm** rằng đôi khi nó khám phá ngẫu nhiên (kể cả vào những hành động tệ) — điều đó không làm lệch giá trị đang học. Đây là lý do $\epsilon_{\min} > 0$ (luôn giữ một xác suất khám phá nhỏ, mãi mãi) **không phá tính tối ưu** của Q-Learning: dù có khám phá ngẫu nhiên tới đâu, target vẫn luôn dùng $\max$, nên $Q$ hội tụ về $Q^*$ chứ không phải về giá trị của chính sách ε-greedy.

SARSA thì khác: vì nó học giá trị của **chính** chính sách ε-greedy đang dùng để hành xử, nếu ε không giảm về 0, SARSA hội tụ về giá trị của chính sách ε-greedy — vốn **không tối ưu**, vì thỉnh thoảng vẫn chọn ngẫu nhiên hành động tệ.

**Ví dụ kinh điển minh hoạ khác biệt (Cliff Walking, Sutton & Barto):** một agent đi trên lưới có một "vực" (đi vào đó bị phạt nặng và về lại vị trí đầu), phải tới đích. Đường đi tối ưu tuyệt đối là đi sát mép vực (ngắn nhất). Nhưng nếu behaviour policy có khám phá ngẫu nhiên, đi sát mép vực rủi ro cao (một bước ngẫu nhiên sai là rơi xuống vực). SARSA (on-policy) học ra một đường đi **an toàn hơn, vòng xa vực** — vì nó tính đến rủi ro của chính hành vi khám phá. Q-Learning (off-policy) học ra đường đi **sát mép, ngắn nhất** — vì nó giả định sẽ luôn hành động tối ưu từ bước tiếp theo trở đi, bất kể thực tế nó có khám phá hay không.

## 4.7 Vì sao learning rate giảm theo từng cặp $(s,a)$ riêng, không theo đồng hồ chung

Tần suất ghé thăm các trạng thái trong blackjack **rất lệch**: `hard 12 vs dealer 6` xuất hiện thường xuyên, còn `soft 21 vs dealer A` (tức bạn có blackjack tự nhiên và dealer ngửa A) hiếm hơn nhiều.

Nếu dùng một đồng hồ $\alpha_t = t^{-\omega}$ chung cho toàn bộ agent (với $t$ = tổng số bước đã train), thì tới khi $t$ đủ lớn, $\alpha_t$ đã rất nhỏ — kể cả với những cặp $(s,a)$ **hiếm** vừa mới được ghé thăm lần đầu vài lần. Bước cập nhật quá nhỏ với dữ liệu quá ít khiến những trạng thái hiếm này **không bao giờ học được đủ**, dù về lý thuyết chúng vẫn được ghé thăm "vô hạn lần" khi $t \to \infty$.

Dùng $\alpha_{N(s,a)} = N(s,a)^{-\omega}$ với $N(s,a)$ là **số lần chính cặp đó** được ghé thăm (không phải tổng số bước toàn cục) đảm bảo mỗi cặp $(s,a)$ có đường cong học riêng, bắt đầu từ $\alpha = 1$ ở lần thăm đầu tiên của **chính nó**, bất kể agent đã train bao lâu tổng thể.

---

# PHẦN 5 — Định lý hội tụ của Q-Learning

## 5.1 Phát biểu định lý (Watkins & Dayan, 1992)

$Q_t \to Q^*$ với xác suất 1, nếu:

1. Mọi cặp $(s,a)$ được ghé thăm **vô hạn lần** (khi $t \to \infty$)
2. **Điều kiện Robbins–Monro:** $\sum_t \alpha_t(s,a) = \infty$ và $\sum_t \alpha_t(s,a)^2 < \infty$
3. Phần thưởng bị chặn (không tiến ra vô cùng)

## 5.2 Trực giác đằng sau điều kiện Robbins–Monro

Điều kiện gồm hai vế đối lập nhau:

**$\sum \alpha_t = \infty$ (tổng phân kỳ)** — nghĩa là tổng "bước đi" theo thời gian là vô hạn, nên dù xuất phát từ ước lượng ban đầu sai bao xa, vẫn còn đủ "quãng đường" tích luỹ để đi tới đáp án đúng. Nếu tổng hội tụ (hữu hạn), bước đi sẽ tắt dần quá nhanh và ước lượng có thể mắc kẹt mãi mãi ở một điểm sai.

**$\sum \alpha_t^2 < \infty$ (tổng bình phương hội tụ)** — nghĩa là bước đi phải nhỏ dần đủ nhanh để **triệt tiêu nhiễu**. Nếu $\alpha_t$ không giảm (hằng số), thì mỗi mẫu ngẫu nhiên mới luôn có ảnh hưởng cỡ cố định lên ước lượng, nên ước lượng sẽ **dao động mãi mãi** quanh giá trị đúng, không bao giờ ổn định tuyệt đối.

**Ghi nhớ ngắn gọn:** *"bước đủ dài để tới được đích, đủ ngắn để tắt được nhiễu."*

## 5.3 Kiểm tra hai vế với $\alpha_n = n^{-\omega}$

Với $\omega = 1$ ($\alpha_n = 1/n$):

$$\sum_{n=1}^{\infty} \frac{1}{n} = \infty \quad \text{(chuỗi điều hoà — phát biểu kinh điển, phân kỳ dù rất chậm)}$$

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6} \approx 1.6449 \quad \text{(hội tụ — bài toán Basel nổi tiếng)}$$

*(Kiểm chứng số: tổng $\sum 1/n$ tới $n=2\times10^6$ đã đạt 15.086 và vẫn đang tăng — rất chậm nhưng không có dấu hiệu chững; tổng $\sum 1/n^2$ tới $n=10^7$ đã đạt 1.644934, khớp $\pi^2/6$ tới 6 chữ số.)*

Vậy $\omega = 1$ thoả cả hai điều kiện. Thực ra bất kỳ $\omega \in (0.5, 1]$ đều thoả (kiểm tra: $\sum n^{-\omega}$ phân kỳ với $\omega \le 1$, và $\sum n^{-2\omega}$ hội tụ với $\omega > 0.5$).

## 5.4 Vì sao $\epsilon_{\min} > 0$ đảm bảo điều kiện 1 mà không phá tính tối ưu

ε-greedy: với xác suất $\epsilon$, chọn hành động **ngẫu nhiên đều** trên mọi hành động khả dĩ; với xác suất $1-\epsilon$, chọn hành động tốt nhất theo ước lượng hiện tại.

Nếu $\epsilon$ giảm về đúng 0, có nguy cơ agent "chốt" vào một chính sách nào đó quá sớm (do ước lượng ban đầu ngẫu nhiên lệch) và **không bao giờ** ghé lại một số cặp $(s,a)$ để sửa sai — vi phạm điều kiện 1.

Giữ $\epsilon_{\min} > 0$ mãi mãi đảm bảo **mọi** cặp $(s,a)$ luôn có xác suất dương được ghé thăm ở mỗi bước, nên theo luật số lớn, chúng được ghé thăm vô hạn lần khi $t \to \infty$.

Vì Q-Learning off-policy (Phần 4.6), việc luôn giữ một chút khám phá ngẫu nhiên **không** làm hỏng target — target luôn dùng $\max$, không phụ thuộc hành vi thật.

## 5.5 Đo được: định luật $n^{-\omega}$ trong thực tế

**Lý thuyết:** với $\alpha_n = n^{-\omega}$, ước lượng $Q_n$ tại một $(s,a)$ cố định là trung bình có trọng số của $n$ mẫu, với "cửa sổ hiệu dụng" (effective window — số mẫu gần nhất đóng góp phần lớn trọng số) có kích thước tỷ lệ $n^\omega$. Nhiễu dư (độ lệch chuẩn của ước lượng quanh giá trị thật) khi đó tỷ lệ $\sigma / \sqrt{\text{cửa sổ}} \sim \sigma n^{-\omega/2}$, nên sai số bình phương trung bình (MSE) tỷ lệ $n^{-\omega}$.

**Cách kiểm chứng SAI** (đã tự mắc và tự sửa): so sánh MSE ở các giá trị $\omega$ khác nhau, cùng một ngân sách $n = 10^6$ ván cố định. Nếu MSE $= n^{-\omega}$ với hằng số chung, tăng $\omega$ từ 0.6 lên 1.0 phải chia MSE cho $10^{6 \times 0.4} \approx 400$ lần. Đo thực tế chỉ khoảng 9 lần.

**Lý do sai:** công thức đầy đủ là $\text{MSE}(n, \omega) = C(\omega) \cdot n^{-\omega}$, trong đó **hằng số $C$ cũng phụ thuộc $\omega$** (vì tốc độ học ban đầu, độ nhiễm bẩn từ bootstrap sớm, v.v. đều khác nhau tuỳ $\omega$). So sánh ngang qua $\omega$ ở cùng một $n$ trộn lẫn hai hiệu ứng (tốc độ hội tụ VÀ hằng số) làm ra kết luận sai.

**Cách kiểm chứng ĐÚNG:** cố định $\omega$, theo dõi MSE khi $n$ tăng dần (không đổi $\omega$). Đo độ dốc của $\log(\text{MSE})$ theo $\log(n)$ — đó chính là số mũ $-\omega$ trong công thức, không bị nhiễm bởi hằng số $C$ (vì $\log$ biến $C \cdot n^{-\omega}$ thành $\log C - \omega \log n$, và độ dốc chỉ là $-\omega$, không phụ thuộc $\log C$).

Đo được: độ dốc **−1.09**, so với lý thuyết **−1.00** ở $\omega = 1$. Đây mới là kiểm chứng thật của định luật.

---

# PHẦN 6 — Đếm bài: biến bài toán dừng thành không dừng

## 6.1 Quá trình dừng (stationary) là gì, và vì sao bộ bài hữu hạn không còn dừng

Một quá trình ngẫu nhiên là **dừng** nếu phân phối xác suất của nó không thay đổi theo thời gian. Với bộ bài vô hạn, $P(\text{lá tiếp theo} = c)$ luôn là $\text{CARD\_PROBS}[c]$, bất kể đã rút bao nhiêu lá trước đó — quá trình dừng.

Với bộ bài hữu hạn, phân phối này **thay đổi** khi bài được rút ra (không hoàn lại) — quá trình **không dừng** (non-stationary).

## 6.2 Trực giác vì sao đếm bài hoạt động

Blackjack có hai bất đối xứng quan trọng liên quan tới lá cao (10, ace) và lá thấp (2–6):

**Lá cao có lợi cho người chơi vì hai lý do:**
1. Nhiều lá 10/A còn lại → nhiều khả năng bạn có blackjack tự nhiên (2 lá đầu cộng 21) → blackjack trả bạn **3:2**, nhưng nếu dealer có blackjack và bạn không, bạn chỉ **thua 1:1** — bất đối xứng này có lợi cho bạn khi xác suất có blackjack tăng.
2. Dealer bị **buộc** rút tới khi đạt ít nhất 17 (không có lựa chọn). Với nhiều lá cao còn lại, dealer dễ quắc hơn khi phải rút thêm.

**Lá thấp có lợi cho dealer:** khi shoe giàu lá thấp, dealer rút thêm ít rủi ro quắc hơn — họ bò lên 17-21 an toàn.

## 6.3 Hệ thống đếm Hi-Lo

$$\text{HILO}(c) = \begin{cases} +1 & c \in \{2,3,4,5,6\} \\ 0 & c \in \{7,8,9\} \\ -1 & c \in \{10, A\} \end{cases}$$

**Running count (RC)** = tổng dồn các giá trị Hi-Lo của mọi lá đã xuất hiện kể từ lần xáo gần nhất.

## 6.4 Vì sao Hi-Lo là hệ "cân bằng" (balanced), và vì sao điều đó quan trọng

Đếm số lá mỗi loại trong một bộ 52 lá: lá thấp (2-6) có $5 \times 4 = 20$ lá (mỗi hạng 4 lá, giá trị $+1$ mỗi lá); lá cao (10, J, Q, K, A) có $16 + 4 = 20$ lá (giá trị $-1$ mỗi lá); lá trung tính (7-9) có $3 \times 4 = 12$ lá (giá trị 0).

Tổng đóng góp: $20 \times (+1) + 20 \times (-1) + 12 \times 0 = 0$.

**Một bộ bài đầy đủ luôn cộng dồn về đúng 0.** Đây là tính chất "cân bằng" của hệ đếm. Nó quan trọng vì: nếu bạn đếm hết một shoe (từ lúc xáo tới lúc dùng hết), running count phải quay về 0. Nếu hệ đếm *không* cân bằng, count sẽ trôi dạt (drift) ngay cả khi shoe hoàn toàn trung tính, và bạn không có cách nào phân biệt "count cao vì shoe thật sự lệch" với "count cao vì hệ đếm tự nhiên trôi dạt".

*(Kiểm chứng: `test_hilo_is_balanced_over_a_full_shoe` xác nhận $\sum \text{HILO}(r) \times \text{CARDS\_PER\_RANK}(r) = 0$ chính xác trên shoe 6 bộ.)*

## 6.5 True Count — chuẩn hoá theo mật độ

$$\text{TC} = \frac{\text{RC}}{\text{số bộ bài còn lại}}$$

**Ví dụ minh hoạ tại sao phải chia:** giả sử RC = +6.

- Nếu còn 5 bộ bài (260 lá): thặng dư 6 lá thấp-hơn-cao bị "dàn mỏng" trên 260 lá còn lại — ảnh hưởng tới lá tiếp theo là rất nhỏ.
- Nếu còn nửa bộ (26 lá): cùng thặng dư 6 đó giờ **tập trung** trên chỉ 26 lá — ảnh hưởng tới lá tiếp theo rất lớn (bạn gần như chắc chắn lá tiếp theo thiên về lá cao).

RC một mình không cho biết **mật độ** thặng dư — nó chỉ cho biết **tổng** thặng dư tuyệt đối. Chia cho số bộ còn lại chuyển nó thành mật độ (một dạng chuẩn hoá tương tự cách bạn nói "nồng độ dung dịch = lượng chất tan / thể tích dung môi", chứ không nói lượng chất tan tuyệt đối).

## 6.6 True Count là nén có mất mát — Effect of Removal

Hi-Lo gán cùng giá trị $+1$ cho mọi lá 2 tới 6. Nhưng về mặt xác suất, việc loại một lá 2 khỏi shoe và loại một lá 6 khỏi shoe **không ảnh hưởng như nhau** tới edge của bạn — lá 6 quan trọng hơn với dealer vì nó là lá dễ khiến dealer quắc nhất khi họ buộc phải rút (nhớ: dealer quắc 42.3% khi ngửa 6, cao nhất trong mọi lá ngửa). Hiện tượng mỗi lá ảnh hưởng edge khác nhau gọi là **effect of removal (EoR)**, và các hệ đếm phức tạp hơn (như hệ đếm "level 2" hay "level 3") gán trọng số tinh vi hơn để nắm bắt sự khác biệt này tốt hơn, đổi lại việc tính toán khó hơn.

Vì Hi-Lo gộp tất cả lá 2-6 vào cùng $+1$, nó **mất thông tin** về lá cụ thể nào đã ra. Nói theo ngôn ngữ Phần 1: true count **không phải** thống kê đủ của shoe — nó chỉ là thống kê **xấp xỉ đủ**. Đây là một đánh đổi có chủ đích: Hi-Lo dễ tính nhẩm tại bàn chơi (chỉ cần cộng/trừ 1), còn hệ đếm chính xác hơn thì khó tính nhẩm hơn nhiều.

## 6.7 Look-ahead bias — giải thích sâu

**Backtesting** (trong tài chính) là việc kiểm tra một chiến lược giao dịch bằng cách "chạy" nó trên dữ liệu lịch sử, xem nó sẽ lãi/lỗ ra sao nếu đã áp dụng trong quá khứ. **Look-ahead bias** là lỗi kinh điển nhất của backtesting: vô tình để chiến lược "nhìn thấy" thông tin mà thực tế nó **chưa có** tại thời điểm ra quyết định (ví dụ dùng giá đóng cửa của ngày để quyết định giao dịch trong chính ngày đó, hoặc dùng báo cáo tài chính đã được "sửa lại" sau này thay vì bản công bố gốc tại thời điểm đó).

Trong dự án này, look-ahead bias xuất hiện nếu bạn đọc true count **sau khi** các lá của ván hiện tại đã được chia, rồi gán **kết quả của ván đó** cho count đã cập nhật. Tại thời điểm đặt cược (trước khi chia bài), người chơi **không thể biết** count sẽ là bao nhiêu sau khi các lá của chính ván đó ra — vì vậy dùng count sau-khi-chia để "giải thích" hay "dự đoán" kết quả của chính ván đó là gian lận thời gian, y hệt backtest dùng giá tương lai.

**Điều làm nó nguy hiểm:** nó không gây lỗi runtime nào. Nó cho ra một đường cong "edge tăng theo count" trông hoàn toàn hợp lý và thuyết phục — chỉ là con số đó **không có ý nghĩa thực tế**, vì không ai có thể tái tạo lại chiến lược đó khi chơi thật (họ không biết trước count-sau-khi-chia lúc đặt cược).

**Cách phòng thủ trong code:** thay vì tin tưởng người lập trình *nhớ* đọc count đúng lúc, hàm `reset()` của môi trường **tự chụp** count ở dòng đầu tiên — trước khi bất kỳ lá nào của ván mới được chia — và trả về giá trị đó qua `info["pre_deal_tc"]`. Người gọi không có cách nào lấy nhầm giá trị, vì họ không bao giờ tự tính count — họ chỉ nhận giá trị đã được đảm bảo đúng thời điểm.

---

# PHẦN 7 — Suy luận thống kê

## 7.1 Định lý giới hạn trung tâm (Central Limit Theorem — CLT)

**Phát biểu (rút gọn, đủ dùng):** nếu $X_1, X_2, \dots, X_n$ là các biến ngẫu nhiên độc lập, cùng phân phối (i.i.d.), có kỳ vọng $\mu$ và phương sai hữu hạn $\sigma^2$, thì khi $n$ lớn, trung bình mẫu

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$$

xấp xỉ phân phối chuẩn: $\bar{X}_n \approx \mathcal{N}(\mu, \sigma^2/n)$.

Điều đáng chú ý: **kết quả này đúng gần như bất kể phân phối gốc của $X_i$ trông thế nào** (có thể rất lệch, rời rạc, kỳ dị) — miễn phương sai hữu hạn và $n$ đủ lớn.

## 7.2 Sai số chuẩn (standard error) và khoảng tin cậy

**Sai số chuẩn của trung bình mẫu:**

$$\text{SE} = \frac{\sigma}{\sqrt{n}}$$

Đây là "độ lệch chuẩn của chính ước lượng trung bình" — càng nhiều mẫu, SE càng nhỏ, theo tốc độ $1/\sqrt{n}$ (giảm chậm: cần **gấp 4 lần** dữ liệu để SE giảm còn **một nửa**).

**Khoảng tin cậy (confidence interval) 95%:**

$$\hat\mu \pm 1.96 \times \text{SE}$$

Con số $1.96$ đến từ phân phối chuẩn: 95% khối lượng xác suất của phân phối chuẩn nằm trong $\pm 1.96$ độ lệch chuẩn quanh trung bình.

**Ví dụ tính tay:** giả sử bạn đo edge $\hat\mu = 0.02$ (2%) từ $n = 1000$ ván, với $\sigma \approx 1$ (độ lệch chuẩn payoff một ván blackjack gần đúng 1):

$$\text{SE} = \frac{1}{\sqrt{1000}} \approx 0.0316$$
$$\text{margin} = 1.96 \times 0.0316 \approx 0.0620$$
$$\text{CI 95\%} = [0.02 - 0.062,\ 0.02 + 0.062] = [-0.042,\ 0.082]$$

*(Kiểm chứng bằng tính toán trực tiếp — khớp chính xác.)*

**Điểm mấu chốt:** dù điểm ước lượng $\hat\mu = 0.02$ là dương, khoảng tin cậy **chứa 0** — nghĩa là với chỉ 1000 mẫu, bạn **không thể** khẳng định tự tin rằng edge thật sự dương. Đây chính xác là lý do dự án cần tới **38.416 ván mỗi bin** (Phần 7.3) để phân biệt edge 1% khỏi 0 một cách đáng tin.

## 7.3 Tính cỡ mẫu cần thiết

Muốn khoảng tin cậy **không chứa 0** khi edge thật sự là $e$, cần margin nhỏ hơn $e$:

$$z \frac{\sigma}{\sqrt{n}} < e \quad\Longrightarrow\quad n > \left(\frac{z\sigma}{e}\right)^2$$

Với $z = 1.96$, $\sigma \approx 1$, $e = 0.01$ (edge 1%):

$$n > \left(\frac{1.96 \times 1}{0.01}\right)^2 = 196^2 = 38{,}416$$

Đây là con số đáng nhớ nhất của phần thống kê: nó giải thích tại sao counter thật cần hàng trăm giờ tại bàn (mỗi giờ chơi được vài chục đến vài trăm ván, mỗi bin count chỉ nhận một phần nhỏ số ván đó), và tại sao mô phỏng cần hàng triệu ván để có kết quả đáng tin ở từng bin.

## 7.4 Sự khác biệt giữa mẫu độc lập và mẫu ghép cặp (paired)

Khi so sánh hai chiến lược A và B, có hai cách thiết kế thí nghiệm:

**Mẫu độc lập (unpaired):** chạy A trên một tập seed, chạy B trên một tập seed **khác**, so sánh trung bình của hai tập kết quả. Phương sai của hiệu số bằng tổng phương sai hai tập (giả sử độc lập):

$$\text{Var}(\bar{A} - \bar{B}) = \text{Var}(\bar{A}) + \text{Var}(\bar{B})$$

**Mẫu ghép cặp (paired):** chạy A và B trên **cùng** một tập seed (cùng thứ tự bài, cùng điều kiện ngẫu nhiên), rồi tính hiệu số **theo từng cặp** trước khi lấy trung bình. Nếu A và B có phần lớn phương sai chung do cùng dùng chung nguồn ngẫu nhiên (cùng bộ bài), phương sai của hiệu số **giảm mạnh**:

$$\text{Var}(\bar{A} - \bar{B})_{\text{paired}} = \text{Var}(\bar{A}) + \text{Var}(\bar{B}) - 2\,\text{Cov}(\bar{A}, \bar{B})$$

Nếu $\text{Cov}(\bar{A}, \bar{B}) > 0$ (điều gần như chắc chắn khi hai chiến lược thấy cùng bộ bài), phương sai ghép cặp **nhỏ hơn** phương sai độc lập.

**Trực giác:** nếu cùng một chuỗi bài "may mắn" (nhiều blackjack tự nhiên) làm cả A và B cùng lãi hơn bình thường, thì so sánh **hiệu** giữa chúng loại bỏ được phần "may mắn chung" đó, chỉ còn lại phần khác biệt thật sự do chiến lược gây ra.

## 7.5 Thống kê t (t-statistic) và ý nghĩa

$$t = \frac{\text{trung bình hiệu số quan sát được}}{\text{sai số chuẩn của hiệu số}}$$

Đây là "hiệu số lớn gấp bao nhiêu lần nhiễu của chính nó". Quy ước phổ biến: $|t| > 2$ (xấp xỉ) thường được coi là "có ý nghĩa thống kê" ở mức tin cậy khoảng 95% (liên hệ tới ngưỡng $1.96$ của CLT).

**Ứng dụng trong dự án:** so sánh đóng góp của "định cỡ cược" và "chiến thuật theo count" bằng thực nghiệm ghép cặp:

| hiệu ứng | chênh lệch trung bình | sai số chuẩn | $t$ |
|---|---|---|---|
| định cỡ cược | +61.45 | 4.80 | **+12.80** |
| chiến thuật theo count | +4.41 | 2.53 | +1.74 |

$t = 12.80$ nghĩa là hiệu ứng lớn gấp gần 13 lần nhiễu của chính nó — **rất** đáng tin. $t = 1.74$ nhỏ hơn ngưỡng 2 thông thường — **không đủ bằng chứng** để khẳng định hiệu ứng khác 0, dù điểm ước lượng là dương.

**Sai lầm cần tránh:** không được tính "tỷ lệ phần trăm đóng góp" (ví dụ "97% từ cược, 3% từ chiến thuật") khi một trong hai số hạng có $t < 2$ — vì bạn đang chia cho (hoặc lấy tỷ lệ của) một đại lượng mà bạn còn chưa chắc **dấu** của nó, chứ đừng nói tới độ lớn chính xác.

---

# PHẦN 8 — Kelly Criterion

## 8.1 Bối cảnh: bài toán tối ưu hoá cỡ cược

Giả sử bạn có một cược lặp lại nhiều lần, mỗi lần bạn chọn stake tỷ lệ $f$ của bankroll hiện tại, và kết quả mỗi lần là payoff ngẫu nhiên $X$ (đơn vị: bội số của stake). Bankroll sau một lần cược:

$$B_{n+1} = B_n (1 + f X_n)$$

Sau $N$ lần cược độc lập cùng phân phối:

$$B_N = B_0 \prod_{n=1}^{N} (1 + f X_n)$$

## 8.2 Vì sao tối ưu hoá lợi nhuận kỳ vọng mỗi vòng là sai

Cách "hiển nhiên" là chọn $f$ tối đa hoá $\mathbb{E}[B_{n+1}] = B_n(1 + f\mu)$ — nếu $\mu > 0$, biểu thức này **tăng vô hạn** khi $f$ tăng, nên "tối ưu" sẽ luôn là cược toàn bộ bankroll ($f=1$) hoặc hơn (đòn bẩy). Nhưng cược toàn bộ mỗi lần gần như chắc chắn dẫn tới phá sản (chỉ cần một lần thua là về 0, và từ 0 không có cách nào phục hồi bất kể edge tốt tới đâu).

Vấn đề: lợi nhuận kỳ vọng **cộng tính** (nếu bạn cộng lãi/lỗ tuyệt đối qua các vòng) không phản ánh đúng bản chất **nhân tính** (multiplicative) của việc cộng dồn bankroll theo thời gian.

## 8.3 Vì sao dùng logarithm

$$\log B_N = \log B_0 + \sum_{n=1}^N \log(1 + fX_n)$$

Log biến **tích** thành **tổng**. Theo Luật số lớn, khi $N$ lớn:

$$\frac{1}{N}\log\frac{B_N}{B_0} = \frac{1}{N}\sum_{n=1}^N \log(1+fX_n) \to \mathbb{E}[\log(1+fX)]$$

Vậy **tốc độ tăng trưởng log dài hạn** hội tụ về $g(f) = \mathbb{E}[\log(1+fX)]$ — một đại lượng **xác định** (không ngẫu nhiên) khi $N \to \infty$. Tối đa hoá $g(f)$ chính là tối đa hoá tốc độ tăng trưởng **thực sự** của bankroll theo thời gian, không phải kỳ vọng cộng tính gây hiểu lầm ở mục 8.2.

**Ví dụ minh hoạ khác biệt cộng/nhân:** lỗ 50% rồi lãi 50% liên tiếp. Cộng tính: $-50\% + 50\% = 0\%$, trông như hoà vốn. Nhân tính thật: $1 \times 0.5 \times 1.5 = 0.75$ — bạn **mất 25%** vốn. Đây gọi là **volatility drag**: biến động (dù trung bình cộng của các % thay đổi là 0) luôn kéo giá trị nhân tính xuống dưới giá trị nếu không có biến động.

## 8.4 Dẫn công thức Kelly bằng khai triển Taylor

Khai triển $\log(1+u)$ quanh $u=0$ bằng chuỗi Taylor:

$$\log(1+u) = u - \frac{u^2}{2} + \frac{u^3}{3} - \frac{u^4}{4} + \dots$$

Với $u = fX$ nhỏ (cược một tỷ lệ nhỏ của bankroll), giữ tới bậc 2:

$$\log(1+fX) \approx fX - \frac{f^2X^2}{2}$$

Lấy kỳ vọng hai vế:

$$g(f) \approx f\mu - \frac{f^2}{2}\mathbb{E}[X^2]$$

Đây là một **hàm bậc hai lõm** theo $f$ (hệ số của $f^2$ âm). Lấy đạo hàm và giải $g'(f) = 0$:

$$g'(f) = \mu - f\,\mathbb{E}[X^2] = 0 \quad\Longrightarrow\quad \boxed{f^* = \frac{\mu}{\mathbb{E}[X^2]}}$$

## 8.5 Vì sao KHÔNG dùng công thức $(bp-q)/b$ phổ biến

Công thức quen thuộc trong sách phổ thông là cho cược **hai kết cục**: thắng $b$ đơn vị (trên mỗi đơn vị cược) với xác suất $p$, thua toàn bộ cược (mất 1 đơn vị) với xác suất $q = 1-p$. Với setup đó:

$$\mu = pb - q(1) = pb - q, \qquad \mathbb{E}[X^2] = p b^2 + q(1)^2$$

Công thức $(bp-q)/b$ chỉ là một cách viết lại **của chính** $f^* = \mu/\mathbb{E}[X^2]$ dưới điều kiện đặc biệt (thường thêm giả định $b=1$ hoặc dùng gần đúng $\mathbb{E}[X^2] \approx b$). Nó **không tổng quát** cho cược nhiều kết cục.

**Blackjack có sáu kết cục khác nhau:** $-2, -1, 0, +1, +1.5, +2$ (thua double, thua thường, hoà, thắng thường, blackjack tự nhiên, thắng double). Ép công thức hai-kết-cục vào đây (chọn $b=1$) sẽ **âm thầm bỏ qua** đóng góp của payoff $1.5$ và $2$ vào cả $\mu$ và $\mathbb{E}[X^2]$, làm sai cả tử số lẫn mẫu số.

**Kiểm tra tính nhất quán (sanity check) — cách rẻ nhất để tự tin công thức tổng quát đúng:** đặt $X = +1$ với xác suất $p$, $X=-1$ với xác suất $q=1-p$ (trường hợp đặc biệt của cược đối xứng, nhị phân). Khi đó:

$$\mu = p(1) + q(-1) = p - q, \qquad \mathbb{E}[X^2] = p(1)^2 + q(-1)^2 = p+q = 1$$

$$f^* = \frac{\mu}{\mathbb{E}[X^2]} = \frac{p-q}{1} = p-q$$

Đây chính xác là công thức Kelly cổ điển cho cược đối xứng nhị phân — công thức tổng quát **thu về** trường hợp quen thuộc, không mâu thuẫn với nó, chỉ mở rộng ra nhiều kết cục hơn.

## 8.6 Đây là xấp xỉ bậc 2 — sai bao nhiêu, và về chiều nào

Số hạng Taylor tiếp theo (bậc 3) mà công thức bậc 2 bỏ qua:

$$g(f) = f\mu - \frac{f^2}{2}\mathbb{E}[X^2] + \frac{f^3}{3}\mathbb{E}[X^3] - \dots$$

Dấu và độ lớn của $\mathbb{E}[X^3]$ (moment bậc ba, liên quan tới **độ xiên/skewness** của phân phối) quyết định xấp xỉ bậc 2 lệch theo chiều nào.

**Đo trên bin true count ≥ 3 của dự án này:**

| | |
|---|---|
| $\mu$ | 0.00890 |
| $\mathbb{E}[X^2]$ | 1.19474 |
| Taylor (bậc 2): $\mu/\mathbb{E}[X^2]$ | 0.00745 |
| Nghiệm số chính xác (tối ưu hoá $g(f)$ trực tiếp bằng số) | **0.00746** |
| $\mathbb{E}[X^3]$ | **+0.213** (xiên dương) |

Xấp xỉ bậc 2 **under-bet 0.1%** so với nghiệm chính xác.

**Vì sao under thay vì over:** vì $\mathbb{E}[X^3] > 0$ (xiên dương — do payoff $+1.5$ blackjack và $+2$ double kéo đuôi phải của phân phối dài hơn đuôi trái), số hạng bậc 3 $\frac{f^3}{3}\mathbb{E}[X^3]$ là **dương**, nên hàm $g(f)$ thật cao hơn xấp xỉ bậc 2 ở vùng $f>0$ một chút, và đỉnh thật của nó nằm ở $f$ **lớn hơn** đỉnh của xấp xỉ bậc 2.

Nếu $\mathbb{E}[X^3] < 0$ (xiên âm — phổ biến ở nhiều loại cược tài chính có "đuôi trái dài", như bán quyền chọn), chiều sẽ **đảo ngược**: xấp xỉ bậc 2 sẽ **over-bet**. Đây là lý do phát biểu chung chung "xấp xỉ Taylor luôn over-bet với phân phối xiên" là **sai một nửa số trường hợp** — phải xét dấu $\mathbb{E}[X^3]$ cụ thể.

## 8.7 Vì sao Fractional Kelly (cược một phần của $f^*$)

Xét lại $g(f)$ chính xác (không xấp xỉ). Đây là hàm **lõm**, đạt cực đại tại $f^*$, và có tính chất đáng chú ý: $g(2f^*) = 0$ — **cược gấp đôi mức Kelly cho tốc độ tăng trưởng dài hạn bằng đúng 0**, dù kỳ vọng cộng tính vẫn dương!

**Vì sao hình phạt bất đối xứng:** vì $g$ lõm và đạt max tại $f^*$, đạo hàm $g'(f^*) = 0$. Gần đỉnh, sai lệch nhỏ theo hướng nào cũng chỉ gây mất mát **bậc hai** (nhỏ) trong tăng trưởng. Nhưng đó chỉ đúng khi sai lệch **nhỏ**. Ở xa đỉnh về phía over-bet (đặc biệt gần $f=2f^*$ trở lên), hàm giảm rất nhanh — thậm chí có thể **âm** (mất tiền chắc chắn về dài hạn) nếu cược đủ quá tay.

**Kết hợp với sai số ước lượng:** trong thực tế bạn không bao giờ biết $\mu$ chính xác — bạn chỉ có $\hat\mu$ ước lượng từ dữ liệu hữu hạn, có nhiễu. Vì $f^* \propto \hat\mu$, nhiễu trong $\hat\mu$ chuyển trực tiếp thành nhiễu trong $\hat{f}$. Do hình phạt bất đối xứng (over-bet tệ hơn under-bet nhiều), **ngay cả khi nhiễu của $\hat\mu$ đối xứng quanh giá trị thật**, kỳ vọng của $g(\hat{f})$ vẫn **thấp hơn** $g(f^*)$ — một dạng "thuế" bắt buộc phải trả cho việc không biết $\mu$ chính xác.

**Cách giảm thiểu:** cược một phân số $\lambda < 1$ của mức Kelly đầy đủ (ví dụ $\lambda = 0.5$, "half Kelly"). Vì hàm $g$ lõm và tương đối phẳng gần đỉnh, giảm $f$ xuống $\lambda f^*$ chỉ mất một phần nhỏ tốc độ tăng trưởng tối đa, nhưng giảm mạnh rủi ro do sai số ước lượng và biến động ngắn hạn.

## 8.8 Xác suất phá sản dạng đóng — dẫn ý tưởng (không chứng minh đầy đủ)

Dưới xấp xỉ khuếch tán (coi quá trình bankroll gần đúng như chuyển động Brown hình học liên tục thời gian — một xấp xỉ hợp lý khi cược rất nhiều lần, mỗi lần nhỏ), có công thức đóng cho xác suất bankroll **từng chạm** một ngưỡng $x$ lần vốn ban đầu, dưới fractional Kelly với hệ số $\lambda$:

$$P(\text{từng chạm } x \cdot B_0) = x^{(2-\lambda)/\lambda}$$

**Tính hai trường hợp cụ thể** ($x = 0.5$, tức "từng mất một nửa vốn"):

- Full Kelly ($\lambda=1$): số mũ $(2-1)/1 = 1$, nên $P = 0.5^1 = 50\%$
- Half Kelly ($\lambda=0.5$): số mũ $(2-0.5)/0.5 = 3$, nên $P = 0.5^3 = 12.5\%$

**Giảm một nửa tỷ lệ cược cắt xác suất mất-nửa-vốn từ 50% xuống 12.5% — giảm bốn lần — trong khi chỉ mất khoảng 25% tốc độ tăng trưởng dài hạn** (có thể tính từ $g(\lambda f^*)$ so với $g(f^*)$ bằng xấp xỉ bậc hai: $g(\lambda f^*) \approx \lambda(2-\lambda) g(f^*)$, với $\lambda=0.5$ cho $0.5 \times 1.5 = 0.75$, tức giữ lại 75% tăng trưởng, mất 25%).

**Giới hạn của công thức này — quan trọng để không lạm dụng:** nó là xấp xỉ khuếch tán, đòi hỏi thời gian liên tục, cược chia nhỏ vô hạn, edge biết chính xác không đổi, và horizon vô hạn. Blackjack thực tế vi phạm cả bốn giả định (số ván rời rạc hữu hạn, cược tối thiểu bắt buộc, edge ước lượng có sai số, horizon hữu hạn trong mọi mô phỏng thực tế). Khi kiểm tra trên dữ liệu mô phỏng của dự án ở horizon 5.000 ván, xác suất phá sản đo được là **0%** ở mọi cấu hình — không mâu thuẫn với công thức, chỉ đơn giản là công thức mô tả một chế độ (nhiều ván, tỷ lệ cược tương đối lớn so với biên độ) khác xa với điều kiện thực tế của mô phỏng này (rất ít ván được cược trên mức tối thiểu, tỷ lệ cược dưới 1% bankroll).

## 8.9 Cổng ý nghĩa thống kê (significance gate)

Ở các bin true count hiếm, $\hat\mu$ ước lượng từ ít mẫu, nên phần lớn giá trị của nó là **nhiễu thống kê** (xem Phần 7). Vì $f^* \propto \hat\mu$, cho Kelly ăn thẳng một $\hat\mu$ nhiễu không chỉ "thêm một chút phương sai vào $f^*$" — nó tạo ra **thiên lệch có hệ thống theo hướng over-bet**, vì bất kỳ khi nào nhiễu ngẫu nhiên đẩy $\hat\mu$ lên cao hơn thực tế, quy tắc cược sẽ **tin tưởng tuyệt đối** vào con số bị thổi phồng đó.

**Quy tắc phòng thủ:** chỉ cược trên mức tối thiểu khi cận dưới của khoảng tin cậy trên $\hat\mu$ vượt quá 0 — nghĩa là edge không chỉ *có vẻ* dương, mà dương **hơn cả sai số đo lường của chính nó**. Đây là kỷ luật tương tự một bàn giao dịch định lượng áp dụng: chỉ đặt cỡ vị thế lớn lên một tín hiệu khi tín hiệu đó **vượt ngưỡng nhiễu ước lượng**, không chỉ khi điểm ước lượng dương.

**Kết quả thực tế:** trong 8 bin true count của dự án, gate chặn **7 bin** — chỉ bin $\geq 3$ có cận dưới CI vượt 0 (edge $+0.890\%$, cận dưới $+0.463\%$). Đáng chú ý, bin $2..3$ có điểm ước lượng **dương** ($+0.267\%$) nhưng cận dưới CI **âm** ($-0.222\%$) — nên đúng theo nguyên tắc, nó **không** được cược, dù trông "có edge".

---

# PHẦN 9 — Đo lường rủi ro

## 9.1 Quy ước dấu: tại sao dùng "loss" thay vì "PnL"

Định nghĩa $L = -\text{PnL}$ (lỗ = âm của lãi/lỗ). Với quy ước này:

- Lãi 30 đơn vị → $\text{PnL} = +30 \to L = -30$
- Lỗ 30 đơn vị → $\text{PnL} = -30 \to L = +30$

**Lý do dùng $L$ thay vì trực tiếp dùng $-\text{PnL}$ khắp nơi:** các báo cáo rủi ro theo thông lệ ngành muốn con số VaR/CVaR **dương** khi có rủi ro thật (dễ đọc: "VaR là 40" nghĩa là "có thể lỗ tới 40", không phải "có thể lãi âm 40" gây khó hiểu). Chốt một quy ước duy nhất, dùng nhất quán, và **test nó** — vì đảo dấu là lỗi phổ biến nhất trong code tính rủi ro, và nó **vô hình**: số vẫn trông hợp lý, chỉ là ý nghĩa bị đảo ngược.

## 9.2 Value at Risk — định nghĩa hình thức

$$\mathrm{VaR}_\alpha(L) = \inf\{x \in \mathbb{R} : P(L \le x) \ge \alpha\}$$

Đây là **định nghĩa dạng phân vị** (quantile): $\mathrm{VaR}_\alpha$ là giá trị nhỏ nhất $x$ sao cho xác suất tích luỹ tới $x$ đạt ít nhất $\alpha$. Nói cách khác, nó là **phân vị thứ $\alpha$** của phân phối lỗ.

**Ví dụ:** $\mathrm{VaR}_{99\%} = 40$ nghĩa là: trên 99% các kịch bản, khoản lỗ **không vượt quá** 40. Ở 1% kịch bản còn lại (kịch bản đuôi), lỗ có thể vượt 40, nhưng VaR không nói **vượt bao nhiêu**.

## 9.3 Conditional Value at Risk (Expected Shortfall)

$$\mathrm{CVaR}_\alpha(L) = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha(L)]$$

Trung bình khoản lỗ, **có điều kiện** rằng bạn đã ở trong vùng đuôi tệ nhất $(1-\alpha)$. Nó trả lời câu hỏi VaR bỏ ngỏ: "nếu tệ, thì tệ **trung bình** tới đâu?"

### Vấn đề với dữ liệu rời rạc — khối xác suất tại VaR

Định nghĩa nêu trên, khi cài đặt ngây thơ (lấy trung bình mọi quan sát $L \ge \mathrm{VaR}$), **sai** khi phân phối rời rạc và có một khối xác suất đáng kể nằm **đúng tại** giá trị VaR.

**Ví dụ minh hoạ (đã kiểm chứng bằng số):** 99 đường mô phỏng hoà vốn ($L=0$), 1 đường cháy tài khoản ($L=100$).

$$\mathrm{VaR}_{90\%} = 0 \quad \text{(phân vị thứ 90 của 100 giá trị, trong đó 99 giá trị là 0)}$$

Cách tính ngây thơ: chọn mọi $L \ge 0$ — tức **cả 100** đường (vì mọi $L$ đều $\ge 0$) — rồi lấy trung bình: $\frac{99 \times 0 + 100}{100} = 1.0$.

Cách đúng: đuôi 10% tệ nhất phải gồm đúng $0.1 \times 100 = 10$ quan sát tệ nhất — tức 9 quan sát giá trị 0 và 1 quan sát giá trị 100. Trung bình đúng: $\frac{9 \times 0 + 100}{10} = 10.0$.

**Sai lệch 10 lần** — không phải sai số nhỏ, mà là đánh giá thấp nghiêm trọng rủi ro đuôi.

### Công thức đúng cho phân phối có khối xác suất

$$\mathrm{CVaR}_\alpha = \frac{1}{1-\alpha}\Big(\underbrace{\mathbb{E}[L \cdot \mathbb{1}\{L > \mathrm{VaR}_\alpha\}]}_{\text{phần thưởng } — \text{ vượt hẳn}} + \underbrace{\mathrm{VaR}_\alpha \cdot \big(P(L \le \mathrm{VaR}_\alpha) - \alpha\big)}_{\text{phần bù — đúng khối xác suất cần}}\Big)$$

**Giải thích từng số hạng:** số hạng đầu lấy trung bình có trọng số của mọi quan sát **thực sự vượt hẳn** VaR (không bao gồm những quan sát *bằng* VaR). Số hạng hai **bù thêm** đúng lượng xác suất còn thiếu (phần khối xác suất nằm chính xác tại VaR) để tổng trọng số đạt đúng $(1-\alpha)$ — không hơn không kém.

Khi không có khối xác suất tại VaR (phân phối liên tục "trơn"), $P(L \le \mathrm{VaR}_\alpha) = \alpha$ chính xác, số hạng hai triệt tiêu về 0, và công thức thu về dạng đơn giản (trung bình những gì vượt VaR). Đây là lý do lỗi này **ẩn được rất lâu**: trên dữ liệu "trông liên tục" (nhiều giá trị khác nhau, không trùng lặp nhiều), công thức sai và công thức đúng cho kết quả gần giống nhau, và chỉ lộ ra khi có khối xác suất đáng kể — như trường hợp payoff blackjack rời rạc với nhiều đường hoà kết quả giống hệt nhau.

## 9.4 Bốn tiên đề của "coherent risk measure"

Artzner, Delbaen, Eber & Heath (1999) đề xuất bốn tính chất mà một thước đo rủi ro $\rho$ "hợp lý" nên có:

1. **Monotonicity:** nếu $X \le Y$ (mọi kịch bản của $X$ đều tệ hơn hoặc bằng $Y$) thì $\rho(X) \ge \rho(Y)$ — vị thế tệ hơn thì rủi ro cao hơn (hoặc bằng).
2. **Translation invariance:** $\rho(X + c) = \rho(X) - c$ với $c$ là hằng số (tiền mặt) — thêm $c$ đơn vị tiền mặt an toàn giảm đúng $c$ đơn vị rủi ro.
3. **Positive homogeneity:** $\rho(\lambda X) = \lambda \rho(X)$ với $\lambda \ge 0$ — nhân đôi quy mô vị thế thì nhân đôi rủi ro (không có hiệu ứng "quy mô lớn thì an toàn hơn theo tỷ lệ" hay ngược lại, trong mô hình cơ bản).
4. **Subadditivity:** $\rho(X+Y) \le \rho(X) + \rho(Y)$ — rủi ro của danh mục gộp không thể lớn hơn tổng rủi ro riêng lẻ. Đây là điều kiện toán học hoá nguyên lý **đa dạng hoá luôn không làm tăng rủi ro**.

## 9.5 Phản ví dụ: VaR vi phạm Subadditivity

**Thiết lập:** hai trái phiếu **độc lập**, mỗi cái vỡ nợ với xác suất 4%, và khi vỡ nợ thì lỗ 100 (khi không vỡ nợ thì lỗ 0).

**VaR riêng lẻ ở mức 95%:** với mỗi trái phiếu, $P(L=0) = 96\% \ge 95\%$, nên $\mathrm{VaR}_{95\%} = 0$ cho **mỗi** trái phiếu riêng lẻ (biến cố vỡ nợ 4% nằm sâu trong đuôi 5%, chưa chạm ngưỡng phân vị 95%).

**VaR của danh mục gộp (cả hai trái phiếu):** xác suất **ít nhất một** trong hai vỡ nợ:

$$P(\text{ít nhất một vỡ nợ}) = 1 - P(\text{không cái nào vỡ nợ}) = 1 - (1-0.04)^2 = 1 - 0.96^2 = 1 - 0.9216 = 0.0784$$

*(Kiểm chứng số: $0.0784 = 7.84\%$.)*

$7.84\% > 5\%$, nghĩa là biến cố "ít nhất một vỡ nợ, lỗ ít nhất 100" nằm **trong** đuôi 5% của phân phối gộp (không còn nằm sâu trong đuôi như từng trái phiếu riêng lẻ). Nên $\mathrm{VaR}_{95\%}(\text{gộp}) = 100$.

**Kiểm tra subadditivity:**

$$\mathrm{VaR}_{95\%}(X+Y) = 100 \quad \text{so với} \quad \mathrm{VaR}_{95\%}(X) + \mathrm{VaR}_{95\%}(Y) = 0 + 0 = 0$$

$$100 > 0 \quad\Longrightarrow\quad \text{VI PHẠM subadditivity}$$

**Ý nghĩa thực tế:** VaR vừa "nói" rằng gộp hai trái phiếu độc lập (đa dạng hoá) làm rủi ro **tăng** từ 0 lên 100 — hoàn toàn ngược với trực giác tài chính (đa dạng hoá nên giảm hoặc ít nhất không tăng rủi ro). Đây là lý do một tổ chức không thể dùng VaR để **cộng gộp** rủi ro giữa các bộ phận/desk một cách đáng tin — tổng VaR các bộ phận không phải là chặn trên hợp lệ cho VaR của toàn tổ chức.

**CVaR không mắc lỗi này** — nó thoả cả bốn tiên đề, đó là lý do các cơ quan quản lý (như trong khung Basel FRTB) chuyển trọng tâm từ VaR sang Expected Shortfall (chính là CVaR) cho mục đích tính vốn dự phòng rủi ro.

## 9.6 Maximum Drawdown

$$\mathrm{MDD} = \max_{t \le T} \frac{M_t - B_t}{M_t}, \qquad M_t = \max_{s \le t} B_s$$

$M_t$ là **đỉnh cao nhất từng đạt được** tính tới thời điểm $t$ (running maximum — cập nhật mỗi khi bankroll đạt mức cao mới). MDD là cú sụt tỷ lệ tệ nhất, tính từ **bất kỳ đỉnh nào** xuống **đáy sau đó**, dọc suốt đường đi.

**Vì sao đo trên toàn bộ đường đi, không chỉ điểm đầu/cuối:** một chiến lược có thể kết thúc ở đúng mức vốn ban đầu (PnL cuối kỳ = 0) nhưng đã từng sụt xuống 10% vốn giữa chừng rồi hồi phục hoàn toàn. Về mặt "kết quả cuối" trông như không rủi ro gì, nhưng thực tế người giữ vị thế đó đã trải qua một giai đoạn cực kỳ căng thẳng, và trong đời thực rất có thể họ đã bị buộc thanh lý (margin call), mất niềm tin và rút lui, hoặc đơn giản không đủ vốn để "chờ hồi phục" — MDD nắm bắt đúng rủi ro đó mà một con số "PnL cuối kỳ" hoàn toàn bỏ sót.

## 9.7 Risk of Ruin

Tỷ lệ đường mô phỏng **từng chạm** một ngưỡng cho trước (ví dụ 50% vốn ban đầu) tại **bất kỳ thời điểm nào** trong suốt quá trình — không chỉ ở cuối.

**Vì sao "từng chạm" (not "kết thúc dưới ngưỡng"):** một đường có thể tụt xuống 40% vốn rồi hồi phục lên 120% vào cuối — về "kết quả cuối" nó thành công, nhưng nó đã **từng** ở trạng thái mà (trong thực tế) người chơi rất có thể đã dừng lại, hết vốn để tiếp tục, hoặc mất bình tĩnh và bỏ cuộc. Tiêu chí "từng chạm" phản ánh rủi ro thực tế đó tốt hơn tiêu chí "kết thúc dưới ngưỡng".

---

# PHẦN 10 — Ba chủ đề nối tất cả lại với nhau

## 10.1 Chủ đề "kiểm chứng độc lập"

Xuyên suốt dự án, mọi khẳng định quan trọng đều được kiểm bằng **ít nhất hai cách tính hoàn toàn không phụ thuộc nhau**:

- EV Phase 1 (đại số thuần) đối chiếu với mô phỏng 300.000 ván (Monte Carlo)
- EV Phase 1 (bộ bài vô hạn) đối chiếu với edge đo trên shoe hữu hạn tại true count = 0 (thống kê thực nghiệm)
- Q-Learning (không biết luật) đối chiếu với Value Iteration (biết chính xác luật)
- Kelly Taylor (bậc 2) đối chiếu với nghiệm số chính xác (tối ưu hoá trực tiếp)
- Công thức risk-of-ruin dạng đóng (xấp xỉ khuếch tán) đối chiếu với đo thực nghiệm (mô phỏng)

Nguyên lý chung: nếu hai phép tính **không chia sẻ giả định hay code** mà vẫn ra cùng kết quả, xác suất cả hai cùng sai theo **đúng cùng một cách** là cực thấp — đó là bằng chứng mạnh. Khi hai phép tính **lệch nhau** (như trường hợp risk-of-ruin), đó cũng là thông tin quý — nó chỉ ra chính xác giả định nào của phép tính lý thuyết bị vi phạm trong thực tế.

## 10.2 Chủ đề "phân biệt số liệu bề mặt với số liệu có ý nghĩa"

Nhiều lần trong dự án, một chỉ số "trông hợp lý" hoá ra không đo đúng thứ cần đo:

- "98.5% ô chính sách đúng" không nói lên **tiền** — phải quy đổi qua `policy_evaluation` ra bps
- "Số ô sai" dao động ít qua các seed, nhưng **chi phí bps** thực sự do những ô đó gây ra lại dao động 5.6 lần — đếm ô là chỉ số nhiễu hơn người ta tưởng
- Trung vị của 4 phân phối độc lập cho **sai dấu** hiệu ứng "chiến thuật theo count" — phải dùng trung bình ghép cặp mới đúng
- "97%/3%" là con số đẹp nhưng **không có ý nghĩa thống kê** vì tính từ một số hạng có $t < 2$

Bài học chung: luôn hỏi "chỉ số này đo đúng cái tôi cần trả lời không, hay chỉ là proxy tiện tính?"

## 10.3 Chủ đề "lỗi nguy hiểm nhất không gây crash"

Bốn lỗi tìm được qua soát xét (tài liệu `05_BON_LOI.md`) đều có chung đặc điểm: **không lỗi nào làm chương trình dừng lại hay báo lỗi.** Chúng đều là lỗi trong *phép tính* hoặc *cách diễn giải*, cho ra những con số trông hoàn toàn hợp lý.

Đây chính là lý do "chạy được, không báo lỗi" **không phải** tiêu chuẩn để tin một kết quả định lượng. Tiêu chuẩn đúng là: **đối chiếu với một phép tính độc lập khác**, hoặc **tính tay một trường hợp nhỏ và so sánh**, hoặc **thử "phá" giả định bằng dữ liệu cố ý được thiết kế để lộ lỗi** (như ví dụ 99 số 0 và 1 số 100 dùng để lộ lỗi CVaR).

# Hiểu code — từng file, cho người mới

Đọc file này cùng với việc mở code thật ra xem. Thứ tự dưới đây là thứ tự **từ dễ đến khó**, không phải thứ tự chữ cái.

---

## Sơ đồ phụ thuộc

```
                    rules.py
                   (luật bài)
                       │
            ┌──────────┼──────────┐
            │          │          │
          hand.py    dp.py    shoe.py
       (luật chơi) (nghiệm    (shoe 6 bộ)
            │       chính xác)     │
      ┌─────┴─────┐              counting.py
      │           │              (Hi-Lo)
   env.py    finite_env.py ─────────┘
 (vô hạn)     (hữu hạn)
      │           │
 qlearning.py  qlearning_count.py
      │           │
      └─────┬─────┘
            │
      simulate.py ──> sizing.py ──> risk.py
                       (Kelly)     (VaR/CVaR)
```

Điều quan trọng: **`dp.py` và `env.py` không biết nhau tồn tại.** `dp.py` tính xác suất bằng tay, `env.py` mô phỏng. Chúng chỉ chung `rules.py`. Đó là lý do khi cả hai ra cùng kết quả thì đó là bằng chứng, không phải lập luận vòng tròn.

---

## 1. `rules.py` (63 dòng) — nền móng

Chỉ có hằng số và 3 hàm thuần. Không class, không trạng thái.

### Xác suất bài

```python
CARD_PROBS = np.array([0.0] + [1/13]*9 + [4/13])
```

`CARD_PROBS[c]` = xác suất rút được lá trị giá `c`. Chỉ số 0 để trống cho `CARD_PROBS[10]` đúng nghĩa "lá 10 điểm".

**Vì sao lá 10 chiếm 4/13?** Vì 10, J, Q, K đều tính 10 điểm — 4 trong 13 hạng bài. Nhầm thành 1/13 là lỗi kinh điển và nó phá toàn bộ kết quả.

### Bộ luật — chỉ định nghĩa ở đây

```python
DEALER_HITS_SOFT_17 = False   # dealer dừng ở mọi 17 (luật S17)
DEALER_PEEKS = True           # dealer lật lá úp khi ngửa A hoặc 10
BLACKJACK_PAYOUT = 1.5        # blackjack trả 3:2
```

Ba dòng này là **toàn bộ** bộ luật. Mọi file khác import về dùng. Đổi `DEALER_HITS_SOFT_17` thành `True` là cả dự án chuyển sang luật H17 và mọi con số đổi theo.

### `add_card(total, usable_ace, card)` — hàm quan trọng nhất file

```python
if card == 1 and total + 11 <= 21:
    total += 11; usable_ace = True     # ace tính 11 nếu không quắc
else:
    total += card                       # ngược lại tính 1
if total > 21 and usable_ace:
    total -= 10; usable_ace = False     # hạ ace từ 11 xuống 1
```

Đây là **nguồn bug số một** trong mọi implementation blackjack. Ba ca phải test:
- `soft 17 + 5` → `hard 12` (không quắc, vì ace hạ xuống)
- `A + A` → `soft 12` (một ace tính 11, một ace tính 1)
- `soft 21 + 10` → `hard 21`

**Vì sao `usable_ace` chỉ là một `bool`, không phải số đếm ace?** Vì nhiều nhất **một** ace có thể tính 11 cùng lúc — hai ace tính 11 đã là 22, quắc rồi. Một bit là đủ.

### `make_rng(seed)`

```python
return np.random.default_rng(seed)
```

Một dòng, nhưng là quyết định thiết kế. **Mọi** nguồn ngẫu nhiên được tạo ở đây và truyền tường minh qua constructor. Không ai gọi `np.random.seed()` toàn cục. Đó là lý do kết quả tái lập bit-identical giữa Windows và Linux.

---

## 2. `hand.py` (131 dòng) — luật chơi dùng chung

File này ra đời sau một vòng soát xét. Trước đó `step()` trùng **19/20 dòng** giữa `env.py` và `finite_env.py`.

Nó chứa những gì **không phụ thuộc vào cách lấy bài**:

| hàm | làm gì |
|---|---|
| `_deal_opening_hand()` | chia 2 lá mỗi bên, xử lý blackjack tự nhiên |
| `_dealer_play()` | dealer rút tới ≥17 (dealer không có lựa chọn) |
| `_showdown(bet)` | so bài, trả +bet / −bet / 0 |
| `step(action)` | áp dụng một hành động |

Class con chỉ cần cung cấp `draw()` (lấy một lá) và `_state()` (trạng thái trả cho agent).

### Về việc khi nào base class là đáng

Phase 1 của dự án này **đã xoá** một abstract base class, vì lúc đó chỉ có một môi trường và lớp trừu tượng chỉ là tầng trung gian vô ích.

Giờ có hai môi trường thật chia sẻ hành vi đáng kể, nên base class **đáng tồn tại**. Nguyên tắc rút ra: **trừu tượng hoá khi có hai ca thật, không phải khi tưởng tượng sẽ có.**

---

## 3. `env.py` (58 dòng) — mô phỏng bộ bài vô hạn

Rất mỏng, vì luật đã ở `hand.py`. Nó chỉ định nghĩa `draw()` và `_state()`.

### `draw()` — lấy mẫu theo lô

```python
if self._buf_i >= self._buf.size:
    u = self.rng.random(self._BUF_SIZE)          # 65.536 số ngẫu nhiên
    self._buf = np.searchsorted(CARD_CDF, u, side="right")
```

Thay vì gọi `rng.choice` cho mỗi lá (chậm vì nó kiểm tra lại vector xác suất mỗi lần), ta sinh 65.536 số ngẫu nhiên đều một lượt rồi tra hàm phân phối tích luỹ. **Cùng phân phối, nhanh hơn nhiều** — nó rút thời gian chạy test từ 24 giây xuống 3 giây.

Kỹ thuật này gọi là **inverse transform sampling**: nếu `U` phân phối đều trên [0,1] và `F` là CDF, thì `F⁻¹(U)` có phân phối của `F`.

---

## 4. `shoe.py` (160 dòng) — shoe 6 bộ

### Vì sao dùng mảng 312 lá, không dùng vector 10 số đếm

Cả hai cho phân phối giống nhau nếu làm đúng. Khác biệt là **loại bug mà mỗi cách cho phép tồn tại**.

Với vector số đếm, bạn phải *tự đảm bảo* không rút lá từ hạng đã hết. Quên một chỗ kiểm tra là chương trình vẫn chạy vui vẻ, rút lá 10 thứ 97 ra khỏi shoe 6 bộ.

Với mảng 312 lá, chuyện đó **không thể xảy ra** — bạn chỉ tăng chỉ số đọc.

> **Nguyên tắc đáng nhớ ngoài dự án này:** ưu tiên thiết kế khiến bug là *bất khả thi*, hơn là thiết kế khiến bug *có thể phát hiện được*.

### Thành phần shoe

```python
CARDS_PER_RANK = np.array([0, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96])
#                          0   1   2   3   4   5   6   7   8   9   10
```

6 bộ × 4 chất = 24 lá mỗi hạng 1–9. Hạng "10" gồm 10/J/Q/K nên 6 × 4 × 4 = 96. Tổng 312.

### `counts` — dữ liệu dẫn xuất

Mảng `counts` (10 số) được cập nhật mỗi lần rút. Nó **không phải nguồn sự thật** — mảng `cards` mới là. Nhưng `counting.py` cần tra nhanh "còn bao nhiêu lá mỗi hạng", và tính lại từ mảng mỗi lần thì tốn.

Vì nó dẫn xuất, ta **assert được** nó luôn khớp với phần còn lại của mảng — và test làm đúng thế.

### `needs_reshuffle()` chỉ *báo*, không xáo

Reshuffle thật xảy ra ở `maybe_reshuffle()`, được gọi **giữa các ván**, không bao giờ giữa ván. Sòng bài thật không xáo giữa ván, và nếu bạn làm vậy thì thành phần bài mà người chơi đang dựa vào bị đổi ngay giữa lúc họ ra quyết định — một dạng look-ahead bias tinh vi.

---

## 5. `counting.py` (200+ dòng) — đếm bài

### Vì sao đếm bài hoạt động

- **Lá cao (10, A) có lợi cho NGƯỜI CHƠI.** Nhiều 10 và A còn lại nghĩa là nhiều blackjack tự nhiên — mà blackjack trả bạn 3:2 nhưng chỉ trả dealer 1:1. Bất đối xứng đó là tiền. Thêm nữa dealer bị buộc rút tới 17 nên quắc nhiều hơn.
- **Lá thấp (2–6) có lợi cho DEALER.** Với lá nhỏ, dealer bò lên 17–21 an toàn thay vì quắc.

### Hi-Lo

```python
#              0   A   2   3   4   5   6   7   8   9  10
HILO = np.array([0, -1, +1, +1, +1, +1, +1, 0, 0, 0, -1])
```

Cộng dồn giá trị này khi lá xuất hiện → **running count**. RC > 0 nghĩa là nhiều lá thấp đã ra hơn lá cao, nên shoe còn lại giàu lá cao: có lợi cho bạn.

**Hi-Lo là hệ "cân bằng":** mỗi bộ có 20 lá thấp (+1) và 20 lá cao (−1), triệt tiêu chính xác. Nên đếm hết một shoe phải về đúng 0. Nếu không cân bằng, count sẽ trôi qua cả một shoe nguyên vẹn và mất ý nghĩa.

### Vì sao chia cho số bộ còn lại

```python
true_count = running_count / decks_remaining
```

Running count +6 có ý nghĩa rất khác nhau tuỳ thời điểm. Còn 5 bộ thì thặng dư 6 lá thấp bị dàn mỏng — gần như không đáng kể. Còn nửa bộ thì thặng dư đó **tập trung**, và lá tiếp theo dễ là lá cao hơn nhiều.

Đây là ý quan trọng nhất của đếm bài, và nó chỉ là **chuẩn hoá**: bạn quan tâm *nồng độ* của thặng dư, không phải độ lớn tuyệt đối. Cùng lý do bạn nói nồng độ dung dịch chứ không nói "tôi bỏ bao nhiêu muối".

### Đây là phép nén CÓ MẤT MÁT — phải nói rõ

Hi-Lo bỏ thông tin. Nó không phân biệt được shoe thiếu năm lá 2 với shoe thiếu năm lá 6, dù hai cái không tốt bằng nhau cho bạn (hiện tượng này gọi là **effect of removal**).

Nên true count **không** phải thống kê đủ (sufficient statistic) của shoe — nó là thống kê *xấp xỉ đủ*. Nói chính xác chỗ này là khác biệt giữa hiểu phương pháp và học thuộc nó.

### `PreDealTracker` và bug nguy hiểm nhất dự án

**Bạn đặt cược TRƯỚC khi bất kỳ lá nào của ván được chia.** Nên count duy nhất được phép dùng để định cỡ cược là count **trước khi chia**.

Nếu bạn đọc count *sau khi* lá của ván đã ra, rồi gán kết quả ván cho count đó, bạn đang dùng thông tin **chưa tồn tại** lúc đặt cược. Đó là **look-ahead bias** — đúng cùng loại lỗi với backtest đọc giá ngày mai.

**Nó không crash. Không raise exception.** Nó tạo ra một đường edge tăng đẹp đẽ, thuyết phục, và hoàn toàn hư cấu.

Phòng thủ ở đây là **cấu trúc, không phải kỷ luật**: môi trường chụp count ở **dòng đầu tiên** của `reset()`, trước khi chia lá nào, và trả về trong `info["pre_deal_tc"]`. Người gọi **không thể lấy sai được**, vì họ không bao giờ tự tính.

### Vì sao lưu histogram đầy đủ, không chỉ (thắng/thua/hoà)

Kelly cần `E[X²]`, không chỉ trung bình. Một ván thắng 1.5 (blackjack) và một ván thắng 2.0 (double) đều là "thắng" nhưng đóng góp rất khác vào phương sai, và do đó vào kích thước cược đúng. Gộp chúng lại là **mất chính thông tin mà việc định cỡ phụ thuộc vào**.

---

## 6. `dp.py` (264 dòng) — nghiệm chính xác

File dài nhất và quan trọng nhất. Nó **không mô phỏng ván nào**.

### Bước 1: `dealer_distribution()` — phân phối kết cục dealer

Dealer không có lựa chọn (luật buộc rút tới ≥17), nên kết cục của họ là phân phối **tính chính xác được** bằng đệ quy có ghi nhớ:

```python
@lru_cache(maxsize=None)
def dealer_distribution(total, usable_ace):
```

Trả về 6-tuple: xác suất dealer kết thúc ở 17, 18, 19, 20, 21, quắc.

Mất **vài mili giây**. Đạt cùng độ chính xác bằng mô phỏng cần cỡ 10⁸ ván.

### Bước 2: `dealer_distribution_from_upcard()` — hàm tinh tế nhất

```python
if DEALER_PEEKS and ((upcard == 1 and hole == 10) or (upcard == 10 and hole == 1)):
    continue          # bỏ qua tổ hợp tạo blackjack
...
return acc / mass     # ← CHIA LẠI XÁC SUẤT
```

Khi dealer ngửa A hoặc 10, họ **đã lật kiểm tra** blackjack trước khi bạn hành động. Nên mọi quyết định của bạn diễn ra trong thế giới "dealer không có blackjack". Phải **loại** các tổ hợp tạo blackjack và **chia lại** cho tổng xác suất còn lại.

**Bỏ dòng `/ mass` đi:** EV lệch ~0.3%, mà bảng chiến thuật vẫn trông hợp lý. Đó chính là lý do nó nguy hiểm — lỗi không lộ ra ở nơi bạn nhìn.

### Bước 3: `stand_values()`

$$V_{\text{stand}}(p, d) = P(\text{dealer quắc} \mid d) + \sum_{k=17}^{21} P(k \mid d)\cdot \text{sign}(p - k)$$

Cài bằng **một phép nhân ma trận**, không vòng lặp Python.

### Bước 4: `value_iteration()`

```python
V = np.zeros((22, 10, 2))
while True:
    hit_v = _hit_backup(V, ...)
    V_new = np.maximum(sv[:, :, None], hit_v)      # ← chữ max
    if np.max(np.abs(V_new - V)) < theta: break
    V = V_new
```

Lặp tới khi ô thay đổi nhiều nhất cũng dưới 10⁻⁹. **13 sweeps.**

### Bước 5: `double_values()` — chỗ dễ sai

```python
succ = np.where(bust[..., None], -1.0, sv[next_t])   # ← sv, KHÔNG PHẢI V
```

Dùng `sv` (giá trị Stand), **không** `V` (giá trị tối ưu). Vì sau khi double bạn **bị cấm rút tiếp**. Dùng nhầm `V` thổi phồng EV, và đây là lỗi rất phổ biến.

Cũng chú ý: double **không nằm trong vòng lặp value iteration**. Vì trạng thái không ghi bạn đã rút mấy lá, nếu nhét vào vòng lặp thì solver sẽ được phép double ở lá thứ ba.

### Bước 6: `policy_evaluation()` — hàm âm thầm và giá trị nhất

```python
V_new = np.where(pi == 1, hit_v, sv[:, :, None])   # ← KHÔNG có max
```

Giống hệt value iteration, chỉ **bỏ chữ `max`** đi. Thay vì "chọn hành động tốt nhất", nó "làm theo chính sách được đưa vào".

Đưa chính sách của agent RL vào → ra EV chính xác của nó → trừ đi EV tối ưu → **2.47 bps**.

Đây là cầu nối biến một chỉ số kỹ thuật ("98.5% ô đúng") thành một con số tiền ("tốn 2.47 điểm cơ bản").

---

## 7. `finite_env.py` (111 dòng) — môi trường bộ bài hữu hạn

Luật chơi giống `env.py` (cùng dùng `hand.py`), khác ở ba chỗ:

1. Bài lấy từ `Shoe`, hết thì xáo lại
2. Duy trì running count Hi-Lo, cập nhật mỗi lần rút
3. `reset()` chụp true count **trước khi chia** và trả trong `info["pre_deal_tc"]`

### Thứ tự dòng trong `reset()` — điều duy nhất phải làm đúng

```python
pre_deal_tc = self.pre_deal_true_count()   # ← TRƯỚC mọi thứ
if self.shoe.maybe_reshuffle():            # ← chỉ giữa các ván
    self.running = 0
info = self._deal_opening_hand()
```

Đảo hai dòng này thì mọi con số Phase 3 và 4 bị nhiễm look-ahead bias, và **không có gì lộ ra**.

### `pre_deal_true_count()` — bug thứ hai cùng loại

Người chơi ở bàn **nhìn thấy** cú xáo. Nên khi shoe đã hết penetration, họ biết count sắp về 0 và phải định cỡ cược theo 0 — không phải theo count cũ đã hết hiệu lực.

Đây là bug tôi tự tìm ra khi viết `bankroll_paths`: nó đọc `env.true_count()` trước khi kiểm tra reshuffle. Không phải look-ahead (dùng thông tin *tương lai*) mà là **stale info** (dùng thông tin *đã hết hạn*). Cả hai đều sai.

---

## 8. `qlearning.py` (200+ dòng) — agent học

Agent **không được cho biết** `CARD_PROBS`, không biết luật dealer, không biết gì. Chỉ chơi và nhận kết quả.

### Công thức cập nhật

$$Q(s,a) \leftarrow Q(s,a) + \alpha_t(s,a)\Big[\underbrace{r + \max_{a'} Q(s',a') - Q(s,a)}_{\text{TD error}}\Big]$$

### Ba chi tiết trong `update()`

```python
self.N[t, up, soft, a] += 1               # đếm lượt thăm, bắt đầu từ 1
alpha = self.N[t, up, soft, a] ** (-self.omega)
target = r if done else r + np.max(self.Q[nt, nup, nsoft])
```

**1. `alpha` tính từ `N[s,a]`, không từ số ván đã chơi.** Tần suất thăm rất lệch — `hard 12 vs 6` xuất hiện nhiều gấp hàng vạn lần `soft 21 vs A`. Dùng đồng hồ chung thì α của trạng thái hiếm tụt về 0 khi chúng còn chưa học được gì.

**2. `target = r` khi `done`.** Không có s' nên không được bootstrap. Sai chỗ này lan ngược qua bootstrap vào gần như mọi trạng thái.

**3. `np.max`** — chữ này khiến Q-Learning là **off-policy**: nó học giá trị của chính sách *tham lam*, bất kể nó *hành xử* thế nào. Đó là lý do ε_min > 0 không phá tính tối ưu. SARSA thay `max` bằng "hành động thực sự chọn" và hội tụ về chỗ khác.

**4. `N^-ω` không phải `(1+N)^-ω`** — đây là một trong 4 lỗi đã sửa. Xem `05_BON_LOI.md`.

### `greedy_policy()` — không được có side effect

```python
coin = np.random.default_rng(0).integers(0, 2, size=q_stand.shape)
```

Dùng generator **cục bộ** có seed cố định, **không** dùng `self.rng`. Vì đây là hàm chỉ đọc — nếu nó tiêu thụ RNG của agent thì việc *quan sát* agent sẽ làm lệch việc *huấn luyện* agent. Đó là lỗi #4 trong `05_BON_LOI.md`.

---

## 9. `sizing.py` — Kelly

$$f^* = \frac{\mu}{\mathbb{E}[X^2]}$$

**Không dùng `(bp−q)/b`** — công thức đó chỉ đúng cho cược **hai kết cục**. Blackjack có sáu: −2, −1, 0, +1, +1.5, +2.

Cách kiểm tra rẻ nhất: với X = ±1, μ = p − q và E[X²] = 1, nên f\* = p − q — đúng công thức Kelly cổ điển. Công thức tổng quát **bao trùm** công thức quen thuộc.

### Điều nghe có vẻ sai

Cho phép Double làm chiến lược **tốt hơn** (−2.42% → −1.09%) nhưng làm `E[X²]` **lớn hơn** (khoảng payoff mở rộng thành ±2). Vì `E[X²]` ở mẫu số, `f*` **giảm** với cùng edge. **Chiến lược mạnh hơn đòi cược tỷ lệ nhỏ hơn** — vì Kelly định giá cả rủi ro, không chỉ kỳ vọng.

### Đây là xấp xỉ bậc 2, không phải nghiệm chính xác

Đo trên bin TC ≥ 3: Taylor cho 0.00745, nghiệm số chính xác cho 0.00746 — xấp xỉ **under-bet 0.1%**. Chiều phụ thuộc dấu moment bậc 3: ở đây E[X³] = +0.213 (xiên dương do +1.5 và +2), nên số hạng bậc 3 làm tăng nghiệm tối ưu.

### Cổng ý nghĩa thống kê

Chỉ cược trên mức tối thiểu nếu **cận dưới khoảng tin cậy của edge vượt 0**. Edge đo từ mẫu hữu hạn thì phần lớn là nhiễu, và Kelly áp vào ước lượng nhiễu **không chỉ thêm phương sai — nó over-bet có hệ thống**.

Thực tế: gate chặn **7 trong 8 bin**. Chỉ bin ≥3 vượt qua.

---

## 10. `risk.py` — đo rủi ro

### Quy ước dấu — chốt một lần và test nó

```python
L = -PnL
```

Lỗ 30 đơn vị → L = +30. Lãi 30 → L = −30. Với quy ước này VaR và CVaR **dương** khi có tiền bị rủi ro, đúng cách báo cáo rủi ro.

Nhầm chiều là bug phổ biến nhất trong code rủi ro, và nó **vô hình** — số vẫn trông như số.

### VaR

$$\mathrm{VaR}_\alpha(L) = \inf\{x : P(L \le x) \ge \alpha\}$$

"VaR 99% là 40 đơn vị" = trên 99% đường, bạn lỗ nhiều nhất 40.

**Điều VaR không cho biết:** chuyện gì xảy ra ở 1% còn lại.

### CVaR

$$\mathrm{CVaR}_\alpha(L) = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha]$$

Trung bình lỗ **khi đã ở trong đuôi xấu**. Nó *thấy* hình dạng đuôi, và đó là lý do cơ quan quản lý chuyển từ VaR sang expected shortfall trong FRTB.

**CVaR ≥ VaR luôn luôn** — đó là đồng nhất thức toán học, nên nó nằm trong code dưới dạng `assert`, không chỉ trong test.

### Vì sao CVaR "coherent" mà VaR không

Bốn tiên đề (Artzner 1999): monotonicity, translation invariance, positive homogeneity, và **subadditivity** — ρ(X+Y) ≤ ρ(X) + ρ(Y).

**VaR vi phạm subadditivity.** Phản ví dụ chuẩn: hai trái phiếu độc lập, mỗi cái vỡ nợ với xác suất 4%, mỗi cái lỗ 100. Ở mức 95%, mỗi cái riêng lẻ có VaR = 0 (vỡ nợ nằm trong đuôi 4% mà VaR 95% không chạm tới). Ghép lại, xác suất ít nhất một cái vỡ nợ ≈ 7.8%, nên VaR 95% của cặp là 100 — lớn hơn 0 + 0.

VaR vừa nói với bạn rằng **đa dạng hoá làm tăng rủi ro**, điều vô nghĩa. Và nó nghĩa là VaR không cộng gộp được giữa các desk.

### Cạm bẫy đã sửa: probability atom

Cách hiển nhiên — lấy trung bình mọi khoản lỗ ≥ VaR — **sai** khi có khối xác suất nằm đúng tại VaR. Xem lỗi #1 trong `05_BON_LOI.md`.

---

## 11. `simulate.py`, `plots.py`, `main.py`

- **`simulate.py`** — chơi ván với một chính sách cho trước, đo edge theo bin, mô phỏng đường bankroll
- **`plots.py`** — 5 hình
- **`main.py`** — chỉ là lớp CLI, không có logic. Đây cũng là lớp **duy nhất không có test**, và đó chính là nơi một bug đã lọt (xem `05_BON_LOI.md` phần cuối)

---

## Bài tự kiểm

Trả lời không nhìn lại. Bí câu nào thì đọc lại mục tương ứng.

1. Vì sao `usable_ace` là `bool` chứ không phải số đếm?
2. Dòng `/ mass` trong `dealer_distribution_from_upcard` làm gì? Bỏ đi thì hỏng thế nào?
3. Vì sao `double_values` dùng `sv` chứ không phải `V`?
4. `policy_evaluation` khác `value_iteration` ở đúng một chữ — chữ nào, và vì sao khác biệt đó quan trọng?
5. Vì sao `alpha` tính từ `N[s,a]` chứ không từ số ván?
6. Chữ `np.max` trong `update` khiến thuật toán off-policy — hệ quả là gì?
7. Vì sao shoe biểu diễn bằng mảng 312 lá chứ không bằng vector 10 số đếm?
8. Vì sao true count phải chia cho số bộ còn lại?
9. Look-ahead bias trong dự án này xuất hiện ở đâu, và phòng thủ bằng cách nào?
10. Vì sao cho phép Double lại làm Kelly fraction *giảm*?
11. Cho phản ví dụ VaR không subadditive.
12. Vì sao `greedy_policy()` không được dùng `self.rng`?

Trả lời được 12/12 là bạn sở hữu dự án này.

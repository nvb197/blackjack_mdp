# Bốn lỗi tìm được qua soát xét

Đây là tài liệu giá trị nhất trong bộ này.

Không phải vì nó cho thấy code hoàn hảo — nó cho thấy code **từng sai**. Giá trị nằm ở chỗ: cả bốn lỗi đều có một điểm chung.

> **Không một lỗi nào làm test đỏ. Không một lỗi nào làm chương trình crash.**

Đó chính là loại lỗi nguy hiểm nhất trong công việc định lượng, và là loại mà một trading desk quan tâm nhất.

---

# LỖI #1 — CVaR bị thiếu hụt khi có khối xác suất tại VaR

## Code sai

```python
v = var(losses, alpha)
tail = losses[losses >= v]
return float(tail.mean())
```

Trông hoàn toàn hợp lý: "CVaR là trung bình các khoản lỗ từ VaR trở lên".

## Vì sao sai

Cách này chỉ đúng cho phân phối **liên tục**. Payoff blackjack là **rời rạc**, nên thường có một khối lớn các đường cùng chung một giá trị.

**Ví dụ cụ thể:** 99 đường hoà vốn (lỗ 0), 1 đường cháy tài khoản (lỗ 100).

| bước | kết quả |
|---|---|
| $\mathrm{VaR}_{90\%}$ | **0** |
| `losses >= 0` chọn ra | **cả 100 đường** |
| `tail.mean()` | **1.0** |
| Sự thật: 10% xấu nhất = 9 số 0 + 1 số 100, trung bình | **10.0** |

**Sai 10 lần.** Code đánh giá thấp đuôi rủi ro một cách nghiêm trọng.

## Công thức đúng

$$\mathrm{ES}_\alpha = \frac{1}{1-\alpha}\Big(\underbrace{\mathbb{E}[L \cdot \mathbb{1}\{L > \mathrm{VaR}\}]}_{\text{mọi thứ vượt hẳn VaR}} + \underbrace{\mathrm{VaR}\cdot\big(P(L \le \mathrm{VaR}) - \alpha\big)}_{\text{đúng phần xác suất tại VaR}}\Big)$$

Số hạng thứ hai cộng lại **chính xác cái sliver** xác suất nằm tại VaR cần thiết để đuôi nặng đúng $(1-\alpha)$ và không hơn.

Khi không có atom, số hạng hai triệt tiêu và công thức thu về dạng đơn giản — đó là lý do bug này **ẩn được rất lâu** trên dữ liệu trông liên tục.

## Một cách fix phổ biến cũng chỉ là xấp xỉ

Cách hay được đề xuất — lấy trung bình $\lceil (1-\alpha)n \rceil$ phần tử tệ nhất — làm tròn kích thước đuôi lên số nguyên, nên bị lệch khi $(1-\alpha)n$ không phải số nguyên.

Đo trên dữ liệu của chính dự án này:

| cách tính | $\mathrm{CVaR}_{99\%}$ |
|---|---|
| trung bình k phần tử tệ nhất | **167.0** |
| **công thức ES chính xác** | **172.7** |

## Tác động thực tế

Trên dữ liệu bankroll của dự án, code cũ **tình cờ cho đúng kết quả** (172.71 = chính xác), vì không có atom nào rơi trúng VaR.

Đó là một **bug tiềm ẩn** — vô hình cho tới khi dữ liệu đổi.

## Test đã thêm

```python
def test_cvar_handles_a_probability_atom_at_var():
    losses = np.array([0.0] * 99 + [100.0])
    assert risk.var(losses, 0.90) == pytest.approx(0.0)
    assert risk.cvar(losses, 0.90) == pytest.approx(10.0)
```

---

# LỖI #2 — Learning rate không phải trung bình mẫu như nó tuyên bố

## Code sai

```python
self.N[t, up, soft, a] += 1
alpha = (1.0 + self.N[t, up, soft, a]) ** (-self.omega)
```

## Vì sao sai

Với $\alpha_n = 1/n$ và $n$ đếm từ 1, cập nhật cho ra **đúng trung bình mẫu**. Lần thăm đầu: $\alpha_1 = 1$, nên $Q_1 = R_1$ — trung bình của một mẫu chính là mẫu đó.

Với $(1+N)^{-1}$: lần thăm đầu $\alpha = 1/2$, nên $Q_1 = R_1/2$. Khai triển đệ quy:

$$Q_1 = \tfrac{1}{2}R_1$$
$$Q_2 = \tfrac{2}{3}Q_1 + \tfrac{1}{3}R_2 = \tfrac{1}{3}(R_1 + R_2)$$
$$Q_3 = \tfrac{3}{4}Q_2 + \tfrac{1}{4}R_3 = \tfrac{1}{4}(R_1+R_2+R_3)$$
$$\vdots$$
$$\boxed{Q_n = \frac{n}{n+1} \times \text{(trung bình mẫu)}}$$

Ước lượng bị **co về 0 vĩnh viễn**, với hệ số chỉ triệt tiêu tiệm cận.

**Kiểm chứng bằng số** (8 phần thưởng, trung bình thật = +0.1875):

| công thức | Q cuối | tỷ lệ so với trung bình thật |
|---|---|---|
| $(1+N)^{-1}$ | +0.166667 | 0.888889 |
| $N^{-1}$ | +0.187500 | **1.000000** |

**Hệ số co theo $n$:**

| n | 10 | 100 | 1.000 | 43.000 |
|---|---|---|---|---|
| $n/(n+1)$ | 0.909 | 0.990 | 0.999 | 0.999977 |

Ở $n = 43.000$ (số lượt thăm điển hình) lệch 0.002% — không đáng kể. Ở $n = 10$ lệch 9% — đáng kể.

## Nghiêm trọng hơn: test đang BẢO VỆ lỗi

```python
def test_terminal_update_targets_the_reward_alone():
    agent.update(15, 3, 0, 0, -1.0, 20, 3, 0, done=True)
    assert agent.Q[15, 3, 0, 0] == pytest.approx(-0.5)   # ← KHOÁ LỖI LẠI
```

Test này khẳng định giá trị **sai** là đúng. Nghĩa là bộ test không những không bắt được lỗi, nó còn **chống lại việc sửa lỗi**.

Đây là một trong những thứ tệ nhất có thể có trong một codebase: một test cho cảm giác an toàn giả.

## Đã sửa

```python
alpha = self.N[t, up, soft, a] ** (-self.omega)
```

Test đổi thành `approx(-1.0)`, cộng thêm hai test mới:
- `test_first_visit_estimate_is_the_observed_reward` — một mẫu phải cho lại chính mẫu đó
- `test_running_estimate_is_the_exact_sample_mean` — cập nhật lặp phải hội tụ về trung bình số học

## Tác động: mọi con số headline đổi

| | trước | sau |
|---|---|---|
| MSE (seed 42, 5M) | 4.48e−05 | **6.47e−05** |
| khớp chính sách | 99.0% | **98.5%** |
| chi phí | 1.70 bps | **2.47 bps** |
| độ dốc log-log | −1.06 | **−1.09** |

## Và một điều bất ngờ

Chạy hai biến thể qua 2 seed ở 1.5 triệu ván: chênh lệch (MSE 1.40e−4 vs 1.34e−4) **nằm trong nhiễu seed** (chi phí dao động 0.00–2.39 bps).

**Cơ chế đáng biết:** ở $\alpha = 1/N$, lần thăm đầu **cam kết toàn bộ** vào target — mà target đó được bootstrap từ bảng Q còn rỗng, nên sai. Việc $(1+N)$ giữ lại chút trọng số của giá trị khởi tạo vô tình hoạt động như **regularizer** chống lại target sớm tệ.

Với trung bình Monte Carlo thuần (chỉ phần thưởng cuối), $1/N$ đúng tuyệt đối. Với **bootstrapping** thì hai cái không tương đương.

Đây cũng là lý do bảng ω giờ **phẳng lại ở đỉnh** (ω = 0.9 ≈ ω = 1.0).

---

# LỖI #3 — `argmax` thiên vị STAND khi hoà

## Code sai

```python
def greedy_policy(self):
    return np.argmax(self.Q, axis=-1)
```

## Vì sao sai

`np.argmax` trả về chỉ số **cực đại đầu tiên**. Ô chưa từng thăm có $Q = [0, 0]$, nên nó luôn trả về 0 = **STAND**.

Agent bị ép một thiên kiến hệ thống: *"cứ không biết thì Stand"* — và đó là thiên kiến nằm trong **báo cáo**, không nằm trong việc học.

## Tác động thực tế: bằng 0

Đo trên run 2 triệu ván:

| | |
|---|---|
| tie trong vùng quyết định (tổng 12–21) | **0** trong 200 ô |
| tie trong toàn bảng | 160 trong 440 |
| ô chưa từng thăm | 160 |
| ô chưa thăm có tổng trong 12–21 | **không có** |

160 ô hoà đều là trạng thái **bất khả thi** (tổng 0, hoặc soft dưới 12).

**Kiểm tra dứt điểm:** đảo hết 160 ô đó từ STAND sang HIT:

```
chi phí với STAND ở ô chưa thăm : 1.7037 bps
chi phí với HIT   ở ô chưa thăm : 1.7037 bps
chênh lệch                      : 0.000000 bps
```

Những ô đó **không thể tới được**, nên không ảnh hưởng bất cứ chỉ số nào.

## Vẫn sửa, vì sao

Một hàm báo cáo không nên mã hoá một sở thích mà dữ liệu không hỗ trợ. Đây là sửa **phòng ngừa**, không phải sửa lỗi đang gây hại.

---

# LỖI #4 — Sửa lỗi #3 tạo ra lỗi lớn hơn

Đây là lỗi tệ nhất trong cả bốn, và nó **do việc sửa lỗi sinh ra**.

## Code sai (bản sửa đầu tiên của lỗi #3)

```python
def greedy_policy(self):
    q_stand, q_hit = self.Q[..., 0], self.Q[..., 1]
    coin = self.rng.integers(0, 2, size=q_stand.shape)   # ← self.rng
    return np.where(q_stand == q_hit, coin, q_hit > q_stand)
```

## Vì sao sai

`self.rng` là generator **của agent**, dùng cho việc chọn hành động khi train. Gọi `.integers()` trên nó **tiêu thụ trạng thái RNG**.

Nhưng `greedy_policy()` là hàm **chỉ đọc**. Nó không được thay đổi trạng thái agent.

## Hậu quả: observer effect

`compare()` gọi `greedy_policy()`. Và `train(eval_every=...)` gọi `compare()` tại mỗi checkpoint.

Nghĩa là **việc quan sát agent làm lệch chính việc huấn luyện agent**:

```
Q giống nhau khi train có/không checkpoint: False
max |chênh lệch| = 0.480992
```

Bảng Q cuối cùng **phụ thuộc vào việc bạn nhìn nó bao nhiêu lần.**

Cụ thể: hình `convergence.png` (sinh với 40 checkpoint) huấn luyện một quỹ đạo **khác** với `main.py ql` (không checkpoint). Hai thứ trông như cùng một thí nghiệm nhưng không phải.

## Vì sao đây là lỗi tệ nhất

Cái thiên lệch mà nó đi sửa đã được đo là **0.000000 bps** — hoàn toàn vô hại.

Tôi đánh đổi một lỗi vô hại lấy một lỗi **phá vỡ tính tái lập** — mà tái lập là cam kết trung tâm của cả dự án.

> **Một observer effect nằm giữa một cam kết về tính tái lập còn tệ hơn cái thiên lệch mỹ phẩm mà nó đi sửa.**

## Đã sửa

```python
coin = np.random.default_rng(0).integers(0, 2, size=q_stand.shape)
```

Generator **cục bộ**, seed cố định. Không đụng `self.rng`, không đụng trạng thái toàn cục, **idempotent** (gọi bao nhiêu lần cũng ra kết quả giống nhau).

## Bốn test đã khoá lại

```python
def test_greedy_policy_does_not_touch_the_agents_rng()
def test_greedy_policy_is_idempotent()
def test_checkpointing_does_not_change_the_training_trajectory()
def test_greedy_policy_does_not_systematically_prefer_stand_on_ties()
```

Test thứ ba là quan trọng nhất: nó train cùng seed **có và không có checkpoint** và đòi bảng Q **giống hệt nhau**.

---

# LỖI #5 (bonus) — test khẳng định một sự tình cờ như thể là bất biến

Sau khi sửa lỗi #2, test hội tụ đỏ. Nguyên nhân đáng chú ý.

## Assertion cũ

```python
for cell in disagreements(agent, pi_star):
    assert cell["gap"] < 0.05
```

Ý tưởng: sai lệch còn lại chỉ nên xảy ra ở nơi hai hành động gần tương đương.

## Vì sao sai

Điều đó **đúng với một lịch trình α và một seed**, và được viết như thể là bất biến.

Nhưng `soft 18 vs 2` có chênh lệch **thật** là **0.0588** — tự nó đã vượt ngưỡng 0.05. Test đòi một điều kiện mà **bản thân trò chơi vi phạm**.

## Đã sửa

Chênh lệch từng ô là **proxy** cho điều thực sự quan trọng — rằng sai lệch **rẻ về tổng thể** — và điều đó đo trực tiếp được:

```python
cost_bps = (dp.expected_value(V_star)
            - dp.expected_value(dp.policy_evaluation(agent.greedy_policy()))) * 10_000
assert 0 <= cost_bps < 10
```

Đo qua 4 seed ở 1 triệu ván: chi phí 1.7–4.0 bps. Ngưỡng 10 bps để chỗ cho biến thiên seed mà vẫn là 4% của house edge 242 bps.

---

# Tổng kết

| # | lỗi | test bắt được? | tác động |
|---|---|---|---|
| 1 | CVaR sai khi có atom | ❌ | tiềm ẩn (0 trên dữ liệu này) |
| 2 | α không phải trung bình mẫu | ❌ **test bảo vệ nó** | mọi số headline đổi |
| 3 | argmax thiên vị STAND | ❌ | 0.000000 bps |
| 4 | greedy_policy tiêu thụ RNG | ❌ | phá tái lập, ΔQ = 0.48 |
| 5 | test khẳng định tình cờ | — | chặn cả việc sửa lỗi #2 |

## Ba bài học

**1. Test có thể bảo vệ lỗi.** Lỗi #2 và #5 đều có test khẳng định hành vi sai là đúng. Một test không tự động là bằng chứng đúng đắn — nó chỉ khoá lại *một hành vi*, và hành vi đó có thể sai.

**2. Sửa lỗi là lúc dễ tạo lỗi.** Lỗi #4 sinh ra **từ việc sửa** lỗi #3, và nó tệ hơn lỗi gốc. Phải audit lại **sau** khi sửa, không chỉ trước.

**3. Loại lỗi nguy hiểm nhất không crash.** Cả bốn đều là lỗi trong *cách tính* hoặc *cách diễn giải*, không phải lỗi cú pháp. Chúng cho ra số trông hợp lý. Cách duy nhất tìm ra chúng là **chạy số và đối chiếu với thứ độc lập**.

---

# Vì sao giữ tài liệu này trong repo

Bản năng đầu tiên là xoá dấu vết và trình bày một repo hoàn hảo.

Đừng làm vậy. Với một dự án ứng tuyển, mục này là phần **mạnh nhất** — không vì nó cho thấy code hoàn hảo, mà vì nó cho thấy **quy trình tự soát xét hoạt động và để lại dấu vết kiểm chứng được.**

Trong phỏng vấn, *"đây là bốn lỗi tôi tìm ra trong code của chính mình, cách tôi tìm ra, và cái thứ tư sinh ra từ việc sửa cái thứ ba"* mạnh hơn nhiều *"code của tôi không có lỗi"* — vì câu thứ hai không ai tin, và nếu họ kiểm thì bạn mất tất cả.

Và nói thật về giới hạn: đã audit 4 vòng, mỗi vòng đều tìm ra thứ mới. Vòng thứ 5 có thể vẫn tìm ra. **Không có trạng thái "chắc chắn không tì vết"** — chỉ có "đã kiểm những thứ tôi nghĩ ra để kiểm".

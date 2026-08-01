# GitHub, phỏng vấn, và việc cần làm tiếp

---

# PHẦN A — Đưa lên GitHub

## A0. Điều kiện tiên quyết

**Đừng push khi chưa làm xong hai việc:**

1. Checkpoint B trong `01_HUONG_DAN_CHAY.md` xanh hết (117 test + 7 lệnh CLI)
2. Đã đọc `02_HIEU_CODE.md` và tự trả lời được ít nhất 8/12 câu ở bài tự kiểm

Push một repo bạn chưa đọc kỹ là tự tạo rủi ro mà không đổi lấy được gì.

## A1. Cài Git

```powershell
git --version
```

Nếu lỗi "not recognized": tải tại **https://git-scm.com/download/win**, cài mặc định, mở lại PowerShell.

## A2. Khai báo danh tính (một lần trên máy)

```powershell
git config --global user.name "Tên bạn"
git config --global user.email "email-github@example.com"
```

Dùng đúng email gắn với tài khoản GitHub.

## A3. Tạo repo trên GitHub (trên trình duyệt)

1. github.com → đăng nhập → nút **+** góc trên phải → **New repository**
2. Tên gợi ý: `blackjack-mdp`
3. Chọn **Private** trước. Đổi sang Public sau khi đã đọc kỹ code.
4. **Không** tick "Add a README", ".gitignore", "license" — để trống
5. **Create repository**

## A4. Khởi tạo git

```powershell
cd E:\Project\blackjack-mdp
git init -b main
```

> Nếu lệnh trên lỗi (Git cũ hơn 2.28), dùng `git init` rồi đổi tên branch **sau** commit đầu tiên bằng `git branch -M main`.

## A5. Nối với GitHub

Thay `TÊN-CỦA-BẠN`:

```powershell
git remote add origin https://github.com/TÊN-CỦA-BẠN/blackjack-mdp.git
```

## A6. Commit theo từng cụm — phần quan trọng nhất

**Đừng commit tất cả một lần.** Một commit duy nhất `"initial commit"` với 3.500 dòng, ngày hôm nay — recruiter đọc `git log` sẽ thấy ngay.

Làm theo 9 cụm dưới đây, mỗi cụm một `add` + một `commit`. Tốt nhất là trải ra vài ngày thật.

**Cụm 1 — nền tảng**
```powershell
git add blackjack/rules.py blackjack/__init__.py tests/__init__.py tests/test_rules.py requirements.txt .gitignore pytest.ini
git commit -m "feat: card rules, hand arithmetic, seeded RNG"
```

**Cụm 2 — luật chơi + môi trường vô hạn**
```powershell
git add blackjack/hand.py blackjack/env.py
git commit -m "feat: shared hand rules and infinite-deck simulator"
```

**Cụm 3 — nghiệm chính xác**
```powershell
git add blackjack/dp.py tests/test_dp.py
git commit -m "feat: exact solution via value iteration"
```

**Cụm 4 — Q-Learning**
```powershell
git add blackjack/qlearning.py tests/test_qlearning.py
git commit -m "feat: tabular Q-learning, validated against the exact solution"
```

**Cụm 5 — shoe hữu hạn**
```powershell
git add blackjack/shoe.py tests/test_shoe.py
git commit -m "feat: six-deck shoe dealt without replacement"
```

**Cụm 6 — đếm bài**
```powershell
git add blackjack/counting.py blackjack/finite_env.py tests/test_counting.py
git commit -m "feat: Hi-Lo counting with look-ahead-safe pre-deal capture"
```

**Cụm 7 — chiến thuật theo count**
```powershell
git add blackjack/qlearning_count.py blackjack/simulate.py
git commit -m "feat: count-augmented agent and simulation harness"
```

**Cụm 8 — Kelly và rủi ro**
```powershell
git add blackjack/sizing.py blackjack/risk.py tests/test_kelly.py
git commit -m "feat: Kelly sizing and risk analytics (VaR, CVaR, drawdown, ruin)"
```

**Cụm 9 — hình, CLI, tài liệu**
```powershell
git add blackjack/plots.py main.py README.md docs figures
git commit -m "docs: figures, CLI, results, and the four bugs found by review"
```

Kiểm tra sau mỗi cụm:
```powershell
git log --oneline
```

## A7. Kiểm không sót file

```powershell
git status
```

**Phải thấy:** `nothing to commit, working tree clean`

## A8. Đẩy lên

```powershell
git push -u origin main
```

Lần đầu GitHub sẽ yêu cầu đăng nhập qua trình duyệt hoặc token.

## A9. Hoàn thiện trên web

- **Add file → Create new file** → tên `LICENSE` → GitHub gợi ý mẫu → chọn **MIT** → điền tên → Commit
- **Settings → Actions** → thêm workflow chạy `pytest -m "not slow"`. Badge "tests passing" tự động đáng giá hơn một câu chữ trong README.

✅ **Checkpoint:** vào trang repo, thấy 14 file trong `blackjack/`, tab **Commits** hiện 9 commit riêng biệt, README hiện đủ 4 hình không vỡ.

---

# PHẦN B — Chuẩn bị phỏng vấn

## B1. Mở đầu 30 giây

> *"Blackjack với bộ bài vô hạn là một MDP hữu hạn nên giải chính xác được bằng quy hoạch động. Em làm điều đó, rồi dùng nghiệm chính xác đó làm chuẩn để kiểm tra một agent Q-Learning không biết luật chơi — nó hội tụ về đúng cùng chính sách, và sai lệch còn lại em đo được là 2.47 điểm cơ bản trên nền house edge 242 bps. Sau đó em mở rộng sang shoe 6 bộ để đo lợi thế của đếm bài, và Kelly để định cỡ tiền cược trên lợi thế đó."*

## B2. Ba câu nên CHỦ ĐỘNG nói ra

**1.** *"Bảng chiến thuật của em khớp published trừ hai ô soft. Một ô chênh 8.6 bps nên coi như hoà; ô kia chênh 74 bps và em vẫn đang tìm nguyên nhân."*
→ Chủ động nêu chỗ lệch mạnh hơn nhiều so với để họ tự tìm ra.

**2.** *"Kelly sizing sẽ vô nghĩa với bộ bài vô hạn — edge giống nhau ở mọi ván nên mọi quy tắc sizing đều thoái hoá thành cược tối thiểu. Phải có shoe hữu hạn trước."*
→ Cho thấy bạn hiểu *tại sao* các phần nằm ở đó, không chỉ làm theo checklist.

**3.** *"Em cố tình không dùng deep RL. State space có 200 phần tử, lập bảng được, nên mạng nơ-ron chỉ thêm sai số xấp xỉ vào bài toán em đã có nghiệm chính xác."*
→ Biết khi nào **không** dùng công cụ mạnh là dấu hiệu trưởng thành.

## B3. Đừng nói

| ❌ | ✅ |
|---|---|
| "Em làm project machine learning" | Nó là **quy hoạch động** trước, RL sau |
| "Agent của em đạt 98.5% accuracy" | Dùng con số **bps** |
| "Em đánh bại được sòng bài" | EV vẫn âm — nói rõ điều đó |
| "Code của em không có lỗi" | "Đây là 4 lỗi em tìm ra và cách tìm" |

## B4. Ngân hàng câu hỏi

Tự trả lời **thành tiếng**, mỗi câu 60–90 giây. Ghi âm lại và nghe.

**MDP và DP**
1. γ = 1 thì toán tử Bellman không còn contraction — sao Value Iteration vẫn hội tụ?
2. Vì sao state space bắt đầu từ tổng 12?
3. Bạn tính P(s'|s,a) thế nào mà không simulate?
4. Vì sao chỉ 13 sweeps? Con số đó nói gì về cấu trúc bài toán?
5. Điều kiện peek là gì? Bỏ nó thì EV lệch bao nhiêu?
6. `double_values` dùng V_stand chứ không phải V* — vì sao?

**Q-Learning**
7. Q-Learning vs SARSA khác nhau chỗ nào? Khi nào chọn cái nào?
8. Ba điều kiện để Q-Learning hội tụ về Q*? Bạn thoả từng cái ra sao?
9. Vì sao ε_min > 0 không phá tính tối ưu?
10. MSE nhỏ mà policy vẫn sai — có thể không? Ngược lại?
11. Vì sao α tính theo N(s,a) chứ không theo đồng hồ toàn cục?
12. Định luật MSE ∝ n^(−ω) — bạn kiểm chứng nó thế nào? Cách nào là sai?

**Đếm bài**
13. Bộ bài hữu hạn có phá vỡ tính Markov không? *(bẫy — cẩn thận)*
14. True count là gì về mặt thống kê?
15. Vì sao không dùng DP cho finite deck?
16. Look-ahead bias xuất hiện ở đâu trong dự án này? Bạn phòng thủ thế nào?
17. Cần bao nhiêu ván để đo được edge 1%? Con số đó nói gì?

**Kelly**
18. Dẫn ra f* = μ/E[X²] trên bảng.
19. Vì sao không dùng (bp−q)/b?
20. Vì sao không full Kelly?
21. Cho phép Double làm f* tăng hay giảm? Vì sao?
22. f* = μ/E[X²] là nghiệm chính xác hay xấp xỉ? Sai bao nhiêu, và về chiều nào?

**Rủi ro**
23. VaR vs CVaR — khác biệt toán học?
24. Cho phản ví dụ VaR không subadditive. *(chuẩn bị vẽ lên bảng)*
25. CVaR của bạn ước lượng từ 400 đường — sai số chuẩn bao nhiêu?
26. Vì sao công thức risk-of-ruin dạng đóng không áp dụng được ở đây?

**Kỹ thuật và tư duy**
27. Bạn biết code đúng nhờ đâu? *(câu trả lời mạnh nhất của bạn)*
28. Kể một lỗi bạn tự tìm ra trong code của mình.
29. Bạn tách đóng góp của định cỡ cược khỏi chiến thuật thế nào?
30. Nếu có thêm một tháng, bạn làm gì?

## B5. Ba câu chuẩn bị kỹ nhất

Ba câu này gần như chắc chắn được hỏi và là nơi bạn ghi điểm nhiều nhất:

- **Câu 1** (γ = 1 và proper policies)
- **Câu 7** (off-policy vs on-policy)
- **Câu 28** (lỗi tự tìm ra) — dùng lỗi #4 trong `05_BON_LOI.md`: *"em sửa một thiên lệch vô hại và tạo ra một observer effect phá tính tái lập"*

---

# PHẦN C — Bullet cho CV

Sau khi đã đọc kỹ và tự tin trả lời được ngân hàng câu hỏi:

> **Optimal Play and Bet Sizing in Blackjack** — Python, NumPy · [github link]
>
> Solved the game exactly as a finite MDP by value iteration (EV −1.087%, matching published figures to five significant figures) and validated it three independent ways: published values, 300k-hand simulation, and convergence of a model-free Q-learning agent.
>
> Quantified the learned policy's residual error in expected value (2.47 bps against a 242 bps house edge) rather than as a share of matching cells, and showed by paired experiment that bet sizing accounts for the gains from card counting (t = 12.8) while count-dependent play does not resolve at this sample size (t = 1.7).
>
> Found and documented four defects that the test suite had not caught — including a unit test that asserted the buggy value and a reproducibility fault introduced by fixing a cosmetic one.

Đoạn thứ ba là đoạn khác biệt nhất. Rất ít CV sinh viên có nó.

---

# PHẦN D — Việc cần làm tiếp, theo thứ tự

## D1. Tuần này

- [ ] Chạy Checkpoint A và B trong `01_HUONG_DAN_CHAY.md`
- [ ] Đọc `02_HIEU_CODE.md`, mở code thật ra đối chiếu
- [ ] Trả lời 12 câu tự kiểm ở cuối `02_HIEU_CODE.md`

## D2. Hai tuần tới — xác lập quyền sở hữu

Chọn **một** file và tự viết lại từ công thức, không nhìn code cũ.

Đề nghị theo thứ tự ưu tiên:
1. **`dp.py`** — bài kiểm tra thật nhất. Tự ra được `-0.02421` là bạn hiểu.
2. **`sizing.py`** — ngắn. Tự dẫn được f* từ đoạn Taylor trong `03_TOAN_HOC.md` là bạn hiểu Kelly.
3. **`shoe.py`** — dễ nhất, nếu bạn cần thắng nhỏ trước.

Chạy test sau khi viết lại. Test xanh = bạn đúng.

## D3. Rồi mới push GitHub

Theo Phần A ở trên.

## D4. Một đóng góp của riêng bạn

Thêm **một** thứ nhỏ không có trong dự án này. Gợi ý theo mức độ:

| việc | độ khó | giá trị |
|---|---|---|
| Test cho từng lệnh CLI (chạy với tham số nhỏ, kiểm exit code 0) | dễ | bịt đúng lỗ đã để lọt một bug |
| So sánh SARSA vs Q-Learning trên cùng môi trường | trung bình | trả lời trực tiếp câu hỏi phỏng vấn #7 |
| Trả lời câu hỏi `soft 13 vs 5` bằng DP có điều kiện trên shoe 6 bộ | khó | đóng lại câu hỏi mở duy nhất của dự án |
| Bootstrap CI cho toàn bộ bảng edge, không chỉ CVaR | trung bình | chặt chẽ hơn về thống kê |

Việc đầu tiên đáng làm nhất vì nó **có động cơ thật**: `main.py` là lớp duy nhất không có test, và đúng ở đó một bug đã lọt.

## D5. Không nên làm

- **Đừng thêm DQN.** State space 200 phần tử. Bạn đã có lý do tốt để từ chối — giữ nó.
- **Đừng thêm Split.** State space bùng nổ, lợi ích cận biên thấp. Nói rõ trong README rằng bạn cố ý bỏ và biết nó đáng ~0.6% EV.
- **Đừng để dự án này ăn vào điểm L3.** Con đường quant ở Pháp gần như bắt buộc đi qua M2 (MASEF, El Karoui, Probabilités et Finance), mà tuyển sinh những chương trình đó **nhìn bảng điểm trước tiên**. Dự án giúp bạn nói chuyện thú vị 15 phút — nó không cứu được điểm trung bình.

## D6. Nhắc về thời gian

Đơn ứng tuyển stage hè 2027 thường mở từ **tháng 10/2026 đến tháng 2/2027**. Nên deadline thật của dự án là **tháng 12/2026**.

Với hồ sơ L3, mục tiêu thực tế là *stage d'assistant* ở phòng risk/middle office, hoặc assistantship nghiên cứu ở Dauphine (CEREMADE) — mà cái sau thì dự án này rất hợp, và **giáo sư của bạn là kênh tốt nhất**, tốt hơn nhiều so với nộp đơn online.

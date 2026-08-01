# Thuật ngữ và khái niệm — tra cứu nhanh

Tài liệu này định nghĩa **mọi** khái niệm kỹ thuật xuất hiện trong dự án — từ lập trình, xác suất thống kê, học tăng cường, tới tài chính định lượng. Dùng như từ điển: gặp một từ lạ ở tài liệu khác, quay lại đây tra.

Sắp xếp theo nhóm chủ đề, trong mỗi nhóm theo thứ tự khái niệm cơ bản trước.

---

# NHÓM A — Lập trình và khoa học máy tính

**Đệ quy (recursion)** — một hàm gọi lại chính nó, với đầu vào "nhỏ hơn" mỗi lần gọi, cho tới khi chạm điều kiện dừng (base case). Ví dụ trong dự án: `dealer_distribution(total, usable_ace)` gọi lại chính nó với `total` mới sau khi rút thêm một lá, cho tới khi `total >= 17` (điều kiện dừng — dealer buộc phải dừng, không tính đệ quy nữa).

**Ghi nhớ / bộ nhớ đệm (memoization)** — kỹ thuật lưu lại kết quả của một lời gọi hàm (theo đầu vào cụ thể), để nếu hàm được gọi lại với **cùng** đầu vào, trả về ngay kết quả đã lưu thay vì tính lại. Trong Python, `@functools.lru_cache` tự động làm việc này. Nó biến một đệ quy có thể tính lại cùng bài toán con nhiều lần (chậm theo cấp số mũ) thành chỉ tính mỗi bài toán con đúng một lần (nhanh tuyến tính hoặc đa thức).

**Vector hoá (vectorization)** — viết một phép tính lặp lại (đáng lẽ cần vòng lặp) dưới dạng một phép toán trên toàn bộ mảng số cùng lúc, để thư viện số học (như NumPy) thực hiện phần lặp thực sự bằng code đã biên dịch sẵn (C/Fortran) thay vì bytecode Python thông dịch từng dòng. Nhanh hơn vòng lặp Python thuần hàng chục tới hàng trăm lần với dữ liệu lớn.

**Broadcasting** (thuật ngữ NumPy) — quy tắc cho phép các phép toán trên mảng có **kích thước khác nhau** tự động "mở rộng" mảng nhỏ hơn để khớp kích thước mảng lớn hơn, mà không cần copy dữ liệu tường minh. Ví dụ: cộng một mảng shape `(10,)` với một mảng shape `(5,10)` sẽ tự động cộng mảng nhỏ vào **từng hàng** của mảng lớn.

**Generator (bộ sinh số ngẫu nhiên)** — một đối tượng giữ trạng thái nội bộ (dựa trên seed ban đầu), mỗi lần được gọi sẽ sinh ra một số "ngẫu nhiên" tiếp theo trong chuỗi, đồng thời cập nhật trạng thái nội bộ để lần gọi sau cho số khác. Trong dự án, `np.random.default_rng(seed)` tạo một generator độc lập, khác với "trạng thái ngẫu nhiên toàn cục" của NumPy (`np.random.seed()`) — dùng generator độc lập giúp tái lập kết quả chính xác mà không lo một phần code khác vô tình làm thay đổi trạng thái ngẫu nhiên dùng chung.

**Seed (hạt giống ngẫu nhiên)** — số nguyên khởi tạo trạng thái nội bộ của một generator. Cùng một seed luôn sinh ra **đúng cùng một chuỗi** số "ngẫu nhiên", cho phép **tái lập** (reproducibility) kết quả — chạy lại đúng cùng thí nghiệm và nhận đúng cùng kết quả, dù bản thân dãy số trông hoàn toàn ngẫu nhiên.

**Trạng thái toàn cục (global state)** — dữ liệu được chia sẻ và có thể bị thay đổi bởi *bất kỳ* phần nào của chương trình, không giới hạn trong một đối tượng cụ thể. `np.random.seed()` thiết lập một trạng thái ngẫu nhiên **toàn cục** — nếu hai phần code khác nhau cùng dùng nó, chúng ảnh hưởng lẫn nhau theo cách khó lường và khó debug. Dự án này tránh hoàn toàn trạng thái toàn cục cho việc sinh số ngẫu nhiên, luôn dùng generator cục bộ truyền tường minh.

**Hiệu ứng phụ (side effect)** — bất kỳ thay đổi nào một hàm gây ra ngoài việc trả về giá trị của nó — ví dụ thay đổi một biến toàn cục, ghi file, hoặc (như trong lỗi #4 của dự án) làm thay đổi trạng thái nội bộ của một đối tượng được truyền vào. Một hàm "chỉ đọc" (không nên có side effect) mà lại có side effect là nguồn bug tinh vi, vì hành vi phụ thuộc vào **thứ tự gọi hàm**, điều rất khó nhận ra khi đọc code.

**Idempotent (tính lặp không đổi)** — một thao tác cho ra **cùng kết quả** dù thực hiện một lần hay nhiều lần liên tiếp (với cùng đầu vào, cùng trạng thái ban đầu). Một hàm không có side effect và không dùng gì ngẫu nhiên mà chưa cố định seed thì tự động idempotent. Dự án sửa `greedy_policy()` để nó idempotent — gọi 10 lần liên tiếp phải ra đúng cùng kết quả.

**Assertion (khẳng định)** — một câu lệnh `assert điều_kiện` yêu cầu chương trình **dừng ngay và báo lỗi** nếu `điều_kiện` sai. Dùng để mã hoá những bất biến (invariant) mà chương trình **luôn phải** thoả mãn nếu logic đúng — ví dụ `assert cvar >= var` (đồng nhất thức toán học, không bao giờ được phép sai). Khác với việc chỉ kiểm tra trong test: assertion nằm ngay trong code sản phẩm, nên nó bắt lỗi **ngay khi chạy thật**, không chỉ khi chạy bộ test.

**Unit test (kiểm thử đơn vị)** — một đoạn code nhỏ, tự động, kiểm tra một hành vi cụ thể của một hàm/class, bằng cách gọi nó với đầu vào đã biết và so kết quả với giá trị mong đợi. Chạy nhanh (mili giây tới vài giây) và chạy lại được bất cứ lúc nào để phát hiện sớm khi một thay đổi code làm hỏng hành vi cũ (gọi là **regression** — hồi quy, nghĩa là quay lại trạng thái tệ hơn).

**Mutation testing (kiểm thử đột biến)** — kỹ thuật đánh giá **chất lượng của chính bộ test**: cố tình sửa (đột biến) một dòng code sản phẩm theo cách làm nó sai, rồi chạy bộ test xem có test nào chuyển sang đỏ (fail) không. Nếu không test nào đỏ, bộ test "có lỗ hổng" — nó không thực sự kiểm tra logic đó, dù bản thân bộ test vẫn "xanh" (pass) trước đột biến. Dự án dùng kỹ thuật này để tự kiểm tra 4 lỗi đã sửa có thực sự được các test mới bắt được không.

**Test có "răng" (test has teeth)** — cách nói không chính thức cho "bộ test thực sự phát hiện được lỗi khi có lỗi", xác nhận qua mutation testing. Đối lập với test "trang trí" — chạy xanh nhưng không kiểm tra gì có ý nghĩa.

**Coverage (độ phủ)** — tỷ lệ dòng code (hoặc nhánh logic) được **chạy tới** ít nhất một lần trong khi chạy bộ test. Coverage cao **không đảm bảo** test tốt — một dòng có thể được chạy tới nhưng kết quả của nó không hề được kiểm tra (không có assertion nào kiểm nó) — đây chính xác là lý do mutation testing bổ sung cho coverage: coverage đo "code có chạy không", mutation testing đo "nếu code sai thì test có phát hiện không".

**Regression test (test hồi quy)** — một test được thêm **sau khi** tìm ra và sửa một lỗi cụ thể, để đảm bảo lỗi đó **không bao giờ quay lại** trong tương lai (nếu ai đó vô tình sửa lại y hệt lỗi cũ). Bốn lỗi trong `05_BON_LOI.md` đều đi kèm regression test tương ứng.

**Docstring** — chuỗi văn bản đặt ngay sau khai báo hàm/class/module trong Python, mô tả nó làm gì, tại sao, và những cạm bẫy cần biết. Khác với comment thông thường, docstring có thể truy cập được bằng code (`help(function)`) và là nơi chuẩn để giải thích *thiết kế*, không chỉ *cú pháp*.

**Refactor (tái cấu trúc)** — thay đổi **cấu trúc nội bộ** của code (cách tổ chức, đặt tên, chia hàm/class) mà **không** thay đổi hành vi bên ngoài (đầu vào/đầu ra vẫn giống hệt). Ví dụ trong dự án: tách `hand.py` ra khỏi `env.py` và `finite_env.py` — code chạy vẫn cho đúng cùng kết quả, chỉ tổ chức gọn hơn.

**Trùng lặp code (code duplication)** — cùng một đoạn logic xuất hiện ở nhiều nơi trong codebase. Nguy hiểm vì khi cần sửa logic đó (ví dụ sửa một luật chơi), người lập trình dễ sửa ở một chỗ mà quên chỗ khác, khiến hai bản "trông giống nhau" âm thầm lệch nhau theo thời gian — đúng vấn đề mà `hand.py` được tạo ra để giải quyết (loại bỏ 19/20 dòng trùng lặp giữa hai môi trường).

**Import chết (dead import / unused import)** — một câu lệnh `import` đưa một tên vào phạm vi (scope) của file nhưng tên đó **không bao giờ được dùng** trong phần còn lại của file. Vô hại về mặt chạy chương trình nhưng là "rác" khiến code khó đọc hơn và có thể gây nhầm lẫn về việc module đó có thực sự cần thiết không.

**Tham số chết (dead parameter)** — một tham số của hàm được khai báo nhưng thân hàm **không bao giờ dùng tới nó**. Thường sót lại sau khi refactor (logic dùng tham số đó đã bị xoá nhưng chữ ký hàm quên cập nhật) — dấu hiệu cần dọn dẹp, và đôi khi là dấu hiệu của một bug tiềm ẩn (tưởng đang dùng nhưng thực ra không).

---

# NHÓM B — Xác suất và Markov Decision Process

**Biến ngẫu nhiên (random variable)** — một đại lượng mà giá trị của nó phụ thuộc vào kết quả của một quá trình ngẫu nhiên, thường ký hiệu bằng chữ hoa ($X$, $L$...).

**Kỳ vọng (expectation / expected value)**, ký hiệu $\mathbb{E}[X]$ — trung bình có trọng số của mọi giá trị khả dĩ của $X$, trọng số là xác suất của từng giá trị: $\mathbb{E}[X] = \sum_x x \cdot P(X=x)$. Trực giác: nếu lặp lại thí nghiệm rất nhiều lần và lấy trung bình cộng các kết quả, con số đó tiến gần $\mathbb{E}[X]$.

**Phương sai (variance)**, ký hiệu $\text{Var}(X)$ hoặc $\sigma^2$ — đo mức độ "phân tán" của $X$ quanh kỳ vọng của nó: $\text{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2]$. **Độ lệch chuẩn** (standard deviation) $\sigma = \sqrt{\text{Var}(X)}$ cùng đơn vị với $X$ (phương sai có đơn vị bình phương), nên dễ diễn giải hơn.

**Moment bậc $k$** — $\mathbb{E}[X^k]$. Moment bậc 1 là kỳ vọng. Moment bậc 2 liên quan tới phương sai ($\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$). Moment bậc 3 liên quan tới **độ xiên** (skewness) — đo mức độ bất đối xứng của phân phối (đuôi phải dài hơn đuôi trái hay ngược lại).

**Xác suất có điều kiện (conditional probability)**, ký hiệu $P(A \mid B)$ — xác suất của $A$, **biết rằng** $B$ đã xảy ra: $P(A \mid B) = P(A \cap B) / P(B)$. Xuất hiện trực tiếp trong dự án ở điều kiện peek (Phần 3.5, tài liệu 03).

**Markov Decision Process (MDP)** — khung toán học mô tả bài toán ra quyết định tuần tự dưới bất định, gồm bốn thành phần: tập trạng thái, tập hành động, hàm chuyển xác suất, hàm phần thưởng. Xem chi tiết Phần 1, tài liệu 03.

**Tính chất Markov (Markov property)** — tương lai chỉ phụ thuộc trạng thái hiện tại, không phụ thuộc lịch sử dẫn tới đó. Xem Phần 1.3, tài liệu 03.

**Thống kê đủ (sufficient statistic)** — một hàm của dữ liệu tóm tắt đủ thông tin cần thiết để suy luận, không mất mát so với dùng toàn bộ dữ liệu gốc. Trong dự án: bộ ba (tổng điểm, lá ngửa, soft/hard) là thống kê đủ của lịch sử ván bài khi bộ bài vô hạn.

**Thống kê xấp xỉ đủ (approximately sufficient statistic)** — một hàm của dữ liệu tóm tắt **phần lớn** thông tin cần thiết nhưng không phải toàn bộ — mất mát một phần. True count là ví dụ: nó nén 10 số đếm shoe thành 1 số, giữ lại phần lớn thông tin liên quan tới edge nhưng mất chi tiết về việc lá cụ thể nào (ví dụ lá 2 hay lá 6) đã ra.

**Chính sách (policy)**, ký hiệu $\pi$ — quy tắc ánh xạ từ trạng thái sang hành động. Xem Phần 1.6, tài liệu 03.

**Hàm giá trị (value function)**, ký hiệu $V(s)$ — kỳ vọng tổng phần thưởng tương lai từ trạng thái $s$. **Hàm giá trị hành động** (action-value function), ký hiệu $Q(s,a)$ — kỳ vọng tổng phần thưởng nếu làm hành động $a$ ở $s$ rồi hành động tối ưu tiếp theo. Xem Phần 2.1–2.2, tài liệu 03.

**Phương trình Bellman** — phương trình đệ quy biểu diễn giá trị của một trạng thái theo giá trị của các trạng thái kế tiếp. Xem Phần 2.3–2.4, tài liệu 03.

**Toán tử Bellman (Bellman operator)** — hàm biến một ước lượng giá trị $V$ thành một ước lượng mới $TV$ bằng cách áp một bước phương trình Bellman. Xem Phần 2.5, tài liệu 03.

**Ánh xạ co (contraction mapping)** — một hàm mà mỗi lần áp dụng làm khoảng cách giữa hai điểm bất kỳ **co lại** theo một tỷ lệ cố định nhỏ hơn 1. Xem Phần 2.5, tài liệu 03.

**Điểm bất động (fixed point)** — một điểm $x$ mà $f(x) = x$ với hàm $f$ cho trước — áp $f$ vào nó không làm nó thay đổi. $V^*$ là điểm bất động của toán tử Bellman.

**Định lý ánh xạ co Banach (Banach fixed-point theorem)** — nếu một hàm là ánh xạ co trên không gian đầy đủ, nó có đúng một điểm bất động, và lặp đi lặp lại hàm đó từ bất kỳ điểm khởi đầu nào cũng hội tụ về điểm đó. Nền tảng lý thuyết cho việc Value Iteration hội tụ.

**Stochastic Shortest Path (SSP)** — lớp bài toán MDP episodic (có kết thúc) mà mọi chính sách đều "proper" (chắc chắn kết thúc trong hữu hạn bước). Blackjack với $\gamma=1$ thuộc lớp này, và đó là lý do nó hội tụ dù mất tính co Banach chuẩn. Xem Phần 2.6, tài liệu 03.

**Chính sách proper (proper policy)** — một chính sách mà dưới nó, xác suất đạt trạng thái kết thúc trong hữu hạn bước là 1 (chắc chắn kết thúc, không "chạy mãi mãi").

**Định lý cải thiện chính sách (Policy Improvement Theorem)** — hành động tham lam theo $Q^\pi$ luôn cho một chính sách mới không tệ hơn $\pi$ ở bất kỳ trạng thái nào. Xem Phần 2.7, tài liệu 03.

---

# NHÓM C — Reinforcement Learning (học tăng cường)

**Reinforcement Learning (RL, học tăng cường)** — nhánh học máy trong đó một "agent" (tác nhân) học cách hành động bằng cách thử và nhận phần thưởng, không được cho biết trước hành động nào đúng (khác với học có giám sát, nơi có nhãn đúng cho từng ví dụ).

**Value Iteration** — thuật toán quy hoạch động giải chính xác MDP khi biết đầy đủ $P(s'|s,a)$, bằng cách lặp áp toán tử Bellman cho tới hội tụ. Xem Phần 3, tài liệu 03.

**Quy hoạch động (Dynamic Programming, DP)** — kỹ thuật giải bài toán bằng cách chia thành các bài toán con chồng lấp, lưu và tái sử dụng kết quả bài toán con thay vì tính lại. Xem Phần 3.2, tài liệu 03.

**Sweep** — một lần duyệt qua toàn bộ tập trạng thái trong Value Iteration, cập nhật giá trị mọi trạng thái một lần.

**Q-Learning** — thuật toán học tăng cường off-policy, học $Q(s,a)$ chỉ từ kinh nghiệm (không cần biết $P(s'|s,a)$), bằng cập nhật temporal-difference. Xem Phần 4, tài liệu 03.

**Temporal-Difference (TD) Learning** — phương pháp cập nhật ước lượng giá trị **ngay sau mỗi bước**, dùng ước lượng của bước kế tiếp làm gần đúng cho phần còn lại (bootstrapping), thay vì đợi kết quả cuối cùng của cả episode như Monte Carlo. Xem Phần 4.2, tài liệu 03.

**Monte Carlo (trong RL)** — phương pháp cập nhật giá trị bằng **kết quả thật sự** của cả episode, sau khi episode kết thúc. Đối lập với TD.

**Bootstrapping (trong RL)** — dùng một ước lượng hiện tại để cập nhật một ước lượng khác (thường là chính nó ở trạng thái trước), thay vì dùng giá trị "sự thật" đã biết chắc chắn. Nguồn gốc tên gọi: "tự kéo dây giày mình lên" — tự cải thiện dựa trên chính ước lượng của mình.

**TD error (sai số temporal-difference)**, ký hiệu $\delta$ — chênh lệch giữa target (ước lượng mới, tốt hơn) và ước lượng cũ. Xem công thức Phần 4.3, tài liệu 03.

**Target (trong TD learning)** — giá trị "gần sự thật hơn" mà bản cập nhật đang cố đưa ước lượng cũ tiến tới. Với Q-Learning, target = phần thưởng thật cộng giá trị tốt nhất ước lượng của trạng thái kế tiếp.

**Learning rate / Step size**, ký hiệu $\alpha$ — tỷ lệ "đi bao xa" về phía target ở mỗi lần cập nhật. $\alpha=1$ nghĩa là nhảy thẳng tới target (bỏ hoàn toàn ước lượng cũ); $\alpha=0$ nghĩa là không cập nhật gì.

**Off-policy** — thuật toán học giá trị của một chính sách (target policy) **khác** với chính sách nó thực sự dùng để thu thập dữ liệu (behaviour policy). Q-Learning là off-policy. Xem Phần 4.6, tài liệu 03.

**On-policy** — thuật toán học giá trị của **chính** chính sách nó đang dùng để hành xử. SARSA là on-policy.

**Behaviour policy** — chính sách thực sự dùng để chọn hành động khi thu thập kinh nghiệm.

**Target policy** — chính sách mà thuật toán đang học giá trị của nó.

**SARSA** — thuật toán on-policy tương tự Q-Learning nhưng target dùng hành động **thực sự được chọn** ở trạng thái kế tiếp (theo behaviour policy), thay vì $\max_{a'}$.

**ε-greedy (epsilon-greedy)** — chiến lược chọn hành động: với xác suất $1-\epsilon$ chọn hành động tốt nhất theo ước lượng hiện tại (khai thác — exploitation), với xác suất $\epsilon$ chọn ngẫu nhiên đều trên mọi hành động (khám phá — exploration).

**Khám phá / Khai thác (exploration / exploitation)** — đánh đổi cơ bản trong RL: khai thác dùng kiến thức hiện có để tối đa hoá phần thưởng ngay bây giờ; khám phá thử hành động chưa chắc tốt để có thể học ra điều gì đó tốt hơn về lâu dài. ε-greedy là cách đơn giản nhất để cân bằng hai điều này.

**Robbins–Monro conditions (điều kiện Robbins–Monro)** — hai điều kiện toán học trên chuỗi learning rate ($\sum \alpha_t = \infty$ và $\sum \alpha_t^2 < \infty$) đảm bảo một quá trình xấp xỉ ngẫu nhiên (như TD learning) hội tụ. Xem Phần 5.1–5.3, tài liệu 03.

**Chuỗi điều hoà (harmonic series)** — $\sum_{n=1}^{\infty} 1/n$, một chuỗi nổi tiếng phân kỳ (tổng tiến ra vô hạn) dù các số hạng tiến về 0 — minh hoạ rằng "các số hạng nhỏ dần" không đủ để đảm bảo tổng hữu hạn.

**Bài toán Basel (Basel problem)** — kết quả cổ điển $\sum_{n=1}^{\infty} 1/n^2 = \pi^2/6$, chứng minh bởi Euler năm 1735. Dùng trong dự án để xác nhận điều kiện Robbins–Monro thứ hai được thoả.

**Bootstrapping bias (thiên lệch do bootstrap)** — sai số hệ thống phát sinh khi các cập nhật TD sớm dùng target được bootstrap từ một ước lượng $Q$ còn rất kém chính xác (gần khởi tạo), khiến "nhiễm bẩn" lan vào các cập nhật sau đó. Giải thích tại sao $\omega=1$ (không giảm trọng số mẫu sớm) không luôn tối ưu trong thực nghiệm của dự án.

**Cliff Walking** — bài toán ví dụ kinh điển (Sutton & Barto) minh hoạ khác biệt Q-Learning vs SARSA: một lưới có "vực", SARSA học đường đi an toàn vòng xa vực, Q-Learning học đường đi tối ưu sát mép vực. Xem Phần 4.6, tài liệu 03.

**Index play** — trong blackjack thực tế, một quyết định chơi bài **phụ thuộc vào true count** thay vì cố định — ví dụ "Stand trên hard 16 vs 10 khi true count ≥ 0" thay vì luôn Hit. Đây là những sai lệch so với basic strategy (chiến thuật cơ bản, không phụ thuộc count) mà một người chơi đếm bài giỏi ghi nhớ thêm.

---

# NHÓM D — Suy luận thống kê

**Định lý giới hạn trung tâm (Central Limit Theorem, CLT)** — trung bình của nhiều mẫu độc lập cùng phân phối, khi số mẫu đủ lớn, xấp xỉ phân phối chuẩn, gần như bất kể phân phối gốc trông ra sao. Xem Phần 7.1, tài liệu 03.

**Phân phối chuẩn (normal / Gaussian distribution)** — phân phối xác suất hình chuông cổ điển, xác định bởi trung bình $\mu$ và độ lệch chuẩn $\sigma$, ký hiệu $\mathcal{N}(\mu, \sigma^2)$.

**Sai số chuẩn (standard error, SE)** — độ lệch chuẩn của chính một **ước lượng** (ví dụ trung bình mẫu), không phải của dữ liệu gốc. $\text{SE} = \sigma/\sqrt{n}$.

**Khoảng tin cậy (confidence interval, CI)** — một khoảng giá trị, tính từ dữ liệu mẫu, được xây dựng sao cho (theo một mức tin cậy cho trước, ví dụ 95%) nó "bẫy" được giá trị thật của tham số đang ước lượng. Không có nghĩa "95% xác suất giá trị thật nằm trong khoảng này" (đó là cách hiểu Bayesian) — nghĩa chính xác hơn (tần suất luận) là: nếu lặp lại thí nghiệm nhiều lần và mỗi lần xây một khoảng như vậy, khoảng 95% số khoảng đó sẽ chứa giá trị thật.

**Cỡ mẫu (sample size)**, ký hiệu $n$ — số lượng quan sát trong một mẫu dữ liệu.

**Mẫu độc lập (unpaired/independent sample)** — hai tập dữ liệu được thu thập độc lập với nhau, không có sự tương ứng theo cặp.

**Mẫu ghép cặp (paired sample)** — hai tập dữ liệu được thu thập sao cho mỗi quan sát trong tập này **tương ứng** với đúng một quan sát trong tập kia (ví dụ cùng seed, cùng bộ bài), cho phép tính hiệu theo từng cặp để loại bỏ phương sai chung. Xem Phần 7.4, tài liệu 03.

**Hiệp phương sai (covariance)**, ký hiệu $\text{Cov}(X,Y)$ — đo mức độ hai biến ngẫu nhiên biến thiên **cùng chiều** với nhau. Dương nếu chúng có xu hướng cùng tăng/giảm, âm nếu ngược chiều, 0 nếu không liên hệ tuyến tính.

**Thống kê t (t-statistic)** — tỷ số giữa hiệu số quan sát được và sai số chuẩn của nó, đo "hiệu số lớn gấp bao nhiêu lần nhiễu của chính nó". Xem Phần 7.5, tài liệu 03.

**Ý nghĩa thống kê (statistical significance)** — một kết quả được coi là "có ý nghĩa thống kê" nếu đủ mạnh (thường đo bằng $|t|$ lớn hoặc khoảng tin cậy không chứa 0) để loại trừ khả năng nó chỉ là do nhiễu ngẫu nhiên, ở một mức tin cậy quy ước (thường 95%).

**Mẫu ngoài (out-of-sample)** — dữ liệu **không** được dùng để hiệu chỉnh (calibrate) một mô hình/tham số, dùng để kiểm tra mô hình có tổng quát hoá tốt không. Đối lập với **trong mẫu (in-sample)** — dữ liệu đã dùng để hiệu chỉnh, nên đánh giá trên chính nó dễ cho kết quả lạc quan giả (overfitting — mô hình "học thuộc" nhiễu của mẫu cụ thể đó thay vì quy luật thật).

**Sai lệch chọn lọc (look-ahead bias)** — lỗi dùng thông tin **chưa tồn tại** tại thời điểm ra quyết định để đánh giá/huấn luyện một chiến lược, khiến kết quả trông tốt hơn thực tế đáng kể. Xem Phần 6.7, tài liệu 03.

**Backtest / Backtesting** — kiểm tra một chiến lược bằng cách mô phỏng nó chạy trên dữ liệu lịch sử.

---

# NHÓM E — Lý thuyết đếm bài (blackjack cụ thể)

**Đếm bài (card counting)** — kỹ thuật theo dõi tỷ lệ lá cao/thấp đã ra khỏi shoe để ước lượng lợi thế hiện tại, từ đó điều chỉnh cách chơi và cỡ cược.

**Hi-Lo** — hệ đếm bài phổ biến nhất, gán $+1$ cho lá 2-6, $0$ cho 7-9, $-1$ cho 10/A. Xem Phần 6.3, tài liệu 03.

**Running count (RC, đếm chạy)** — tổng dồn giá trị Hi-Lo của mọi lá đã xuất hiện kể từ lần xáo gần nhất.

**True count (TC, đếm thật)** — running count chia cho số bộ bài còn lại trong shoe, chuẩn hoá theo mật độ. Xem Phần 6.5, tài liệu 03.

**Hệ đếm cân bằng (balanced counting system)** — hệ đếm mà tổng giá trị trên toàn bộ một bộ bài chuẩn bằng đúng 0. Xem Phần 6.4, tài liệu 03.

**Effect of Removal (EoR, hiệu ứng loại bỏ)** — hiện tượng mỗi lá bài cụ thể ảnh hưởng edge của người chơi một lượng khác nhau khi bị loại khỏi shoe, dù các hệ đếm đơn giản (như Hi-Lo) gộp nhiều lá vào cùng một giá trị. Xem Phần 6.6, tài liệu 03.

**Penetration (độ thấm/độ sâu)** — tỷ lệ số lá đã dùng trên tổng số lá của shoe, tại thời điểm sòng bài xáo lại. Penetration 75% nghĩa là dùng 75% shoe rồi mới xáo (còn 25% "để dành", không dùng tới).

**Shoe** — bộ bài gộp từ nhiều bộ 52 lá tiêu chuẩn (thường 6-8 bộ), xáo chung, dùng cho nhiều ván liên tiếp trước khi xáo lại.

**Basic strategy (chiến thuật cơ bản)** — chính sách chơi tối ưu, **không phụ thuộc** true count, tính cho một bộ luật cụ thể. Trong dự án, đây chính là $\pi^*$ tính ở Phase 1.

**Index play** — xem lại ở Nhóm C, đây là biến thể của basic strategy có phụ thuộc true count.

**Wonging** — thuật ngữ lóng trong giới đếm bài (đặt theo tên Stanford Wong) chỉ hành vi **chỉ tham gia bàn/đặt cược khi true count thuận lợi**, và rời bàn/ngồi ngoài khi count bất lợi — thay vì buộc phải cược liên tục mọi ván.

---

# NHÓM F — Kelly Criterion và lý thuyết cược

**Kelly Criterion** — quy tắc xác định tỷ lệ bankroll tối ưu để cược, nhằm tối đa hoá tốc độ tăng trưởng log dài hạn của bankroll. Xem Phần 8, tài liệu 03.

**Bankroll** — tổng số vốn hiện có dành cho việc cược.

**Payoff** — kết quả (lãi/lỗ) của một lần cược, tính theo bội số của số tiền đã cược.

**Tốc độ tăng trưởng (growth rate)**, trong bối cảnh Kelly — tốc độ tăng trưởng log dài hạn của bankroll, $g(f) = \mathbb{E}[\log(1+fX)]$.

**Volatility drag (kéo lùi do biến động)** — hiện tượng biến động (dù trung bình cộng bằng 0) luôn làm giảm giá trị tích luỹ nhân tính so với trường hợp không biến động. Ví dụ: −50% rồi +50% cho kết quả −25%, không phải 0%. Xem Phần 8.3, tài liệu 03.

**Khai triển Taylor (Taylor expansion / Taylor series)** — biểu diễn một hàm phức tạp gần một điểm bằng tổng các số hạng đa thức bậc tăng dần, mỗi số hạng liên quan tới đạo hàm bậc tương ứng tại điểm đó. Dùng để dẫn công thức Kelly gần đúng $f^* = \mu/\mathbb{E}[X^2]$. Xem Phần 8.4, tài liệu 03.

**Xấp xỉ bậc 2 (second-order approximation)** — xấp xỉ giữ tới số hạng bậc 2 của khai triển Taylor, bỏ qua các bậc cao hơn.

**Độ xiên (skewness)** — thước đo độ bất đối xứng của một phân phối xác suất, liên quan tới moment bậc 3. Xiên dương: đuôi phải dài hơn (các giá trị cực lớn hiếm nhưng có). Xiên âm: đuôi trái dài hơn.

**Fractional Kelly** — cược một tỷ lệ $\lambda < 1$ của mức Kelly đầy đủ, để giảm rủi ro do sai số ước lượng, đổi lấy giảm một phần tốc độ tăng trưởng tối đa. Xem Phần 8.7, tài liệu 03.

**Xấp xỉ khuếch tán (diffusion approximation)** — mô hình gần đúng một quá trình rời rạc, nhiều bước nhỏ bằng một quá trình liên tục thời gian (chuyển động Brown), hợp lý khi số bước lớn và mỗi bước nhỏ. Dùng để dẫn công thức đóng cho risk-of-ruin. Xem Phần 8.8, tài liệu 03.

**Cổng ý nghĩa thống kê (significance gate)** — quy tắc chỉ áp dụng cỡ cược lớn khi tín hiệu (edge ước lượng) vượt ngưỡng nhiễu ước lượng của chính nó (ví dụ cận dưới khoảng tin cậy > 0). Xem Phần 8.9, tài liệu 03.

---

# NHÓM G — Đo lường rủi ro

**PnL (Profit and Loss)** — lãi (dương) hoặc lỗ (âm) của một vị thế/chiến lược trong một khoảng thời gian.

**Quy ước dấu (sign convention)** — thoả thuận về việc một đại lượng dương/âm nghĩa là gì (ví dụ $L = -\text{PnL}$ để lỗ luôn dương). Xem Phần 9.1, tài liệu 03.

**Phân vị (quantile)** — giá trị mà một tỷ lệ $p$ cho trước của phân phối nằm **dưới hoặc bằng** nó. Phân vị thứ 50 là trung vị (median).

**Value at Risk (VaR)** — phân vị thứ $\alpha$ của phân phối lỗ: mức lỗ không bị vượt quá với xác suất $\alpha$. Xem Phần 9.2, tài liệu 03.

**Conditional Value at Risk (CVaR) / Expected Shortfall (ES)** — trung bình khoản lỗ, có điều kiện đã ở trong vùng đuôi tệ nhất $(1-\alpha)$. Xem Phần 9.3, tài liệu 03.

**Khối xác suất / atom (probability atom)** — một giá trị cụ thể mà phân phối xác suất gán một xác suất dương **không nhỏ tuỳ ý** cho nó (khác với phân phối liên tục, nơi mọi điểm cụ thể có xác suất bằng 0). Phân phối payoff blackjack có nhiều atom (ví dụ, rất nhiều đường mô phỏng cùng kết thúc ở đúng giá trị lỗ 0 — hoà vốn).

**Coherent risk measure (thước đo rủi ro chặt chẽ)** — một thước đo rủi ro thoả bốn tiên đề: monotonicity, translation invariance, positive homogeneity, subadditivity. Xem Phần 9.4, tài liệu 03.

**Subadditivity (tính dưới cộng tính)** — tính chất $\rho(X+Y) \le \rho(X)+\rho(Y)$: rủi ro gộp không vượt tổng rủi ro riêng. VaR vi phạm tính này; CVaR thoả mãn. Xem Phần 9.5, tài liệu 03.

**Maximum Drawdown (MDD)** — cú sụt tỷ lệ tệ nhất từ đỉnh xuống đáy sau đó, dọc suốt một đường bankroll. Xem Phần 9.6, tài liệu 03.

**Running maximum (đỉnh chạy)** — giá trị lớn nhất đã đạt được tính tới thời điểm hiện tại, cập nhật liên tục khi có giá trị mới cao hơn.

**Risk of Ruin (xác suất phá sản)** — xác suất bankroll từng chạm một ngưỡng cho trước (ví dụ 50% vốn ban đầu) tại bất kỳ thời điểm nào. Xem Phần 9.7, tài liệu 03.

**Bootstrap (thống kê, khác với bootstrapping trong RL)** — kỹ thuật ước lượng độ bất định (ví dụ khoảng tin cậy) của một thống kê bằng cách lấy mẫu **có hoàn lại** nhiều lần từ chính dữ liệu quan sát được, tính lại thống kê đó trên mỗi mẫu lấy lại, rồi xem sự phân tán của các kết quả đó. Không liên quan tới "bootstrapping" trong TD learning (Nhóm C) dù trùng tên gốc tiếng Anh — hai khái niệm hoàn toàn khác nhau, chỉ tình cờ chia sẻ ẩn dụ "tự dựa vào chính mình".

**FRTB (Fundamental Review of the Trading Book)** — khung quy định vốn dự phòng rủi ro của Uỷ ban Basel, chuyển trọng tâm đo lường rủi ro từ VaR sang Expected Shortfall (CVaR).

---

# NHÓM H — Các khái niệm dùng chung/liên ngành

**Cơ bản (fundamental) vs Kỹ thuật (technical)** — không dùng trong dự án này nhưng thường gặp cùng lĩnh vực; không cần định nghĩa ở đây.

**Basis point (bps, điểm cơ bản)** — đơn vị bằng $1/100$ của $1\%$, tức $0.0001$. Dùng phổ biến trong tài chính để tránh mơ hồ khi nói về thay đổi phần trăm nhỏ. Xem thêm cuộc trò chuyện trước trong dự án.

**Đánh đổi bias–variance (bias-variance tradeoff)** — nguyên lý chung trong ước lượng thống kê: một ước lượng có thể "thiên lệch nhưng ổn định" (bias cao, variance thấp) hoặc "không thiên lệch nhưng nhiễu" (bias thấp, variance cao), và thường không thể giảm cả hai cùng lúc mà không đánh đổi. Xuất hiện trong dự án ở việc chọn độ rộng bin true count (bin hẹp: variance cao trên mỗi bin; bin rộng: bias vì gộp các count khác giá trị lại với nhau).

**Ground truth (sự thật nền)** — kết quả đúng, đã biết chắc chắn, dùng làm chuẩn để so sánh/kiểm tra một phương pháp ước lượng hay học máy khác. Trong dự án, nghiệm chính xác từ Value Iteration ($V^*, \pi^*$) chính là ground truth để kiểm Q-Learning.

**Kiểm chứng độc lập (independent verification)** — kiểm tra một kết quả bằng một phương pháp tính toán **hoàn toàn khác**, không chia sẻ code hay giả định với phương pháp gốc, để giảm khả năng cả hai cùng mắc chung một lỗi.

**Đường cơ sở (baseline)** — kết quả của phương án đơn giản nhất/mặc định, dùng làm điểm so sánh cho các phương án phức tạp hơn (ví dụ "flat betting" là baseline để so sánh với Kelly).

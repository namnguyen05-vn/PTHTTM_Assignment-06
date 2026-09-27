"""Theory of a simple recurrent network, with formulas and executable examples."""
from report_engine import *
def theory():
    page("CHƯƠNG 2. DỮ LIỆU CHUỖI VÀ HÀM HỒI TIẾP",chapter=2)
    p("Dữ liệu chuỗi là một dãy quan sát có thứ tự. Vị trí của một phần tử mang ý nghĩa bên cạnh giá trị của nó. Hai phiên truy cập cùng có một lần xem và một lần thêm giỏ hàng nhưng khác thứ tự có thể biểu thị các tình huống khác nhau. Tương tự, một chuỗi tăng rồi giảm giá không tương đương với chuỗi giảm rồi tăng.")
    math(r"X=(x_1,x_2,\ldots,x_T),\quad x_t\in\mathbb{R}^{D}")
    table(["Ký hiệu","Ý nghĩa"],[["T","Số bước thời gian trong cửa sổ"],["D","Số đặc trưng ở mỗi bước"],["H","Số chiều trạng thái ẩn"],["B","Số mẫu trong minibatch"]],[90,393],caption="Các chiều tensor trong mô hình chuỗi.")
    p("Bước thời gian là vị trí quan sát, không mặc định là một khoảng thời gian vật lý cố định. Retailrocket có khoảng cách giữa các sự kiện không đều; chứng khoán được quan sát theo phiên giao dịch. Việc đưa khoảng cách thời gian vào đặc trưng không biến RNN thành mô hình thời gian liên tục, nhưng cung cấp thêm thông tin về nhịp độ hành vi.")
    p("Một mẫu huấn luyện trong bài là một cửa sổ lịch sử và một nhãn tương lai. Mã khách hàng hoặc mã cổ phiếu xác định ranh giới chuỗi; không được nối hai thực thể chỉ vì chúng xuất hiện cạnh nhau trong tệp CSV.")

    page("2.2. Nơ-ron, phép affine và tanh")
    p("Phép affine kết hợp tuyến tính đầu vào rồi cộng độ lệch. Một hàm kích hoạt phi tuyến được áp dụng sau đó để mở rộng lớp hàm mà mô hình có thể biểu diễn. Nếu nhiều lớp chỉ gồm phép affine, hợp thành của chúng vẫn là một phép affine.")
    math([r"z=xW+b",r"\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},\quad \tanh'(z)=1-\tanh^2(z)"])
    code("""z = np.array([-2., 0., 2.])
    activation = np.tanh(z)
    derivative = 1 - activation**2
    ""","Giá trị và đạo hàm của tanh.")
    p("Tanh đưa mỗi thành phần trạng thái vào khoảng (-1,1). Ở vùng gần không, hàm phản ứng tương đối mạnh với thay đổi đầu vào; ở vùng bão hòa, đạo hàm gần không. Giới hạn biên độ trạng thái không đồng nghĩa giới hạn toàn bộ gradient theo tham số, vì đường đạo hàm còn đi qua các ma trận trọng số.")
    p("Trong dữ liệu nhiều thang đo, chuẩn hóa đầu vào giúp tránh một số biến số lớn áp đảo ngay từ phép affine đầu tiên. Thống kê chuẩn hóa được ước lượng trên giai đoạn train; dùng cả test để ước lượng sẽ đưa thông tin phân phối tương lai vào quá trình xây dựng mô hình.")

    page("2.3. Trạng thái ẩn và phép truy hồi")
    p("RNN cơ bản duy trì một vector trạng thái ẩn để biểu diễn lịch sử đã quan sát. Trạng thái mới phụ thuộc đồng thời đầu vào hiện tại và trạng thái trước. Đây là cơ chế hồi tiếp: đầu ra trung gian ở một bước trở thành đầu vào của bước kế tiếp [1].")
    math([r"h_0=0,\quad h_t=\tanh(x_tW_x+h_{t-1}W_h+b_h)",r"z=h_TW_y+b_y"])
    table(["Tham số","Kích thước","Vai trò"],[["W_x","D × H","Biến đổi quan sát hiện tại"],["W_h","H × H","Biến đổi trạng thái trước"],["b_h","H","Độ lệch của trạng thái"],["W_y, b_y","H × C; C","Ánh xạ trạng thái sang đầu ra"]],[90,115,278],caption="Các tham số theo quy ước vector hàng.")
    p("Trạng thái ẩn không phải bản sao toàn bộ dữ liệu quá khứ và không có nghĩa mỗi chiều tương ứng một khái niệm quan sát được. Nó là biểu diễn số học được học để giảm hàm mất mát. Với H=32, toàn bộ lịch sử trong cửa sổ được tổng hợp thành 32 giá trị trước đầu dự đoán.")
    p("Trong bài, h₀ được đặt bằng không ở mỗi cửa sổ. Trạng thái không được truyền tùy tiện từ minibatch trước sang minibatch sau, vì các mẫu trong batch có thể thuộc những thực thể khác nhau.")

    page("2.4. RNN như hợp thành hàm theo thời gian")
    math(r"h_T=f_{x_T}\circ f_{x_{T-1}}\circ\cdots\circ f_{x_1}(h_0)")
    p("Với một quan sát x cố định, f_x là hàm biến đổi trạng thái. RNN áp dụng nhiều lần cùng quy tắc tham số hóa trên các quan sát khác nhau. Đồ thị khi được trải theo thời gian có nhiều bước, nhưng các bước dùng chung W_x, W_h và b_h.")
    code("""def recurrent_step(x, h, wx, wh, bias):
        return np.tanh(x @ wx + h @ wh + bias)

    h = np.zeros(hidden_size)
    for x_t in sequence:
        h = recurrent_step(x_t, h, wx, wh, bias)
    ""","Vòng lặp thể hiện phép hợp thành theo thời gian.")
    p("Chia sẻ trọng số giữ số tham số độc lập với độ dài T. Tuy nhiên, chi phí tính toán vẫn tăng theo T vì trạng thái tại bước sau cần trạng thái của bước trước. Khác với một phép tổng các quan sát, thứ tự thực hiện các hàm nói chung ảnh hưởng kết quả.")
    p("Trải đồ thị là cách diễn giải cùng một mô hình, không tạo ra T bộ trọng số độc lập. Nếu mỗi vị trí có tham số riêng, mô hình sẽ có giả định và khả năng xử lý độ dài khác hẳn RNN được sử dụng ở đây.")

    page("2.5. Ví dụ số của chuỗi ba bước")
    code("""x = np.array([[1., 0.], [0., 1.], [1., 1.]])
    wx = np.array([[.3, -.2], [.1, .4]])
    wh = np.eye(2) * .5
    bias = np.zeros(2)
    h = np.zeros(2)
    states = [h.copy()]
    for x_t in x:
        h = np.tanh(x_t @ wx + h @ wh + bias)
        states.append(h.copy())
    ""","Ví dụ hai chiều để quan sát trạng thái từng bước.")
    import numpy as np
    x=np.array([[1.,0.],[0.,1.],[1.,1.]])
    wx=np.array([[.3,-.2],[.1,.4]]);wh=np.eye(2)*.5;h=np.zeros(2);rows=[["0","0,0000","0,0000"]]
    for i,a in enumerate(x,1):
        h=np.tanh(a@wx+h@wh);rows.append([i,dec(h[0]),dec(h[1])])
    table(["Bước","h₁","h₂"],rows,[90,196,197],caption="Trạng thái tính trực tiếp từ ví dụ NumPy.")
    p("Ở bước thứ hai, đầu vào chỉ kích hoạt đặc trưng thứ hai nhưng trạng thái vẫn chịu ảnh hưởng của bước thứ nhất qua h₁W_h. Khi đảo thứ tự ba vector đầu vào, phép cộng trạng thái xảy ra ở các thời điểm khác nhau trước tanh, nên trạng thái cuối thường thay đổi.")
    p("Ví dụ này không chứa bước học: trọng số được ấn định để làm rõ phép tính. Trong thực nghiệm, gradient điều chỉnh trọng số dựa trên sai khác giữa dự đoán và nhãn. Phân biệt phép suy luận với quá trình học giúp tránh đồng nhất một ví dụ số với một mô hình đã được huấn luyện.")

    page("2.6. Many-to-one và đầu dự đoán")
    p("Cả hai nhiệm vụ dùng cấu trúc many-to-one: nhiều bước lịch sử tạo một đầu ra. Retailrocket có ba logits cho ba hành vi; chứng khoán có một đầu ra thực cho log return của phiên kế tiếp.")
    math([r"p_c=\frac{\exp(z_c)}{\sum_j\exp(z_j)}",r"\widehat r_{t+1}=s_r\,z,\quad \widehat P_{t+1}=P_t\exp(\widehat r_{t+1})"])
    table(["Khía cạnh","Hành vi","Chứng khoán"],[["Đầu ra","3 logits","1 giá trị thực"],["Biến đổi","Softmax","Nhân độ lệch chuẩn train"],["Quyết định","Argmax","Dự báo log return"],["Diễn giải thêm","Xác suất từng hành vi","Giá đóng cửa hàm ý"]],[120,181,182],caption="Hai đầu dự đoán dùng chung lõi hồi tiếp.")
    p("Logits là điểm số chưa chuẩn hóa, không phải xác suất. Softmax chỉ áp dụng cho bài phân loại. Đầu hồi quy không dùng softmax vì tổng bằng một và miền giá trị bị ràng buộc sẽ không phù hợp với lợi suất âm hoặc dương.")
    p("Dự báo giá được xây dựng từ giá đã biết ở cuối cửa sổ và return dự báo. Khi return dự báo bằng không, giá dự báo bằng giá gần nhất. Do đó đồ thị giá có thể trông gần đúng ngay cả khi mô hình chưa vượt được đối chứng đơn giản.")

    page("2.7. Padding, masking và độ dài thật")
    p("Các phiên khách hàng có độ dài khác nhau. Để gom chúng thành tensor B×T×D, chương trình đặt chuỗi thật ở đầu và thêm số không vào phần còn lại. Lengths lưu số bước thật; phần đệm được bỏ qua bằng masking trong Keras và packed sequence trong PyTorch.")
    code("""mask = np.arange(max_length)[None, :] < lengths[:, None]
    x[~mask] = 0.0
    # Keras receives mask; PyTorch receives packed lengths.
    ""","Mặt nạ phần đệm theo độ dài thật.")
    p("Số không sau chuẩn hóa cũng có thể là một giá trị dữ liệu hợp lệ. Vì thế mask được xây dựng từ lengths, không suy ra bằng điều kiện mọi đặc trưng bằng không. Điều này tách rõ dữ liệu thật khỏi ô trống phục vụ tính toán.")
    table(["Sai sót","Hệ quả"],[["Lấy trạng thái sau toàn bộ padding","Mô hình tiếp tục biến đổi lịch sử bằng bước giả"],["Che theo giá trị bằng 0","Có thể bỏ nhầm một bước thật"],["Đệm trái nhưng xử lý như đệm phải","Độ dài và vị trí quan sát không khớp"]],[220,263],caption="Các lỗi thường gặp khi xử lý chuỗi ngắn.")
    p("Kiểm tra độc lập thay phần đệm từ 0 thành 10.000 và yêu cầu dự đoán giữ nguyên. Bài kiểm tra này đánh giá hành vi của mô hình, bổ sung cho việc kiểm tra hình dạng tensor.")

    page("2.8. Cross-entropy và trọng số lớp")
    math([r"\ell_i=-w_{y_i}\log p_{i,y_i}",r"L_B=\frac{1}{|B|}\sum_{i\in B}\ell_i"])
    p("Trong Retailrocket, hành vi xem chiếm phần lớn dữ liệu. Cross-entropy có trọng số tăng đóng góp của lớp hiếm vào gradient. Trọng số được tính từ nhãn train theo căn bậc hai của nghịch đảo tần suất, rồi chuẩn hóa để trung bình theo mẫu bằng một.")
    code("""counts = np.bincount(y_train, minlength=3)
    weight = np.sqrt(counts.sum() / (3*np.maximum(counts, 1)))
    weight /= np.dot(counts, weight) / counts.sum()
    loss = (per_sample_cross_entropy * weight[y]).mean()
    ""","Trọng số chỉ dựa trên phân bố lớp train.")
    p("Trọng số không tạo thêm thông tin cho lớp hiếm và không bảo đảm recall tăng trong mọi lượt. Nó thay đổi mục tiêu tối ưu. Chỉ số validation và test được tính trên phân bố quan sát gốc; không nhân trọng số lại vào accuracy hoặc confusion matrix.")
    p("Hai framework đều lấy trung bình loss đã nhân trọng số theo số mẫu trong batch. Quy ước này được ghi rõ vì một số API loss có trọng số có thể dùng mẫu số khác, dẫn đến hai chương trình trông tương tự nhưng cập nhật gradient không tương đương.")

    page("2.9. Hồi quy log return và sai số")
    math([r"r_{t+1}=\log(P_{t+1}/P_t)",r"L_B=\frac{1}{|B|}\sum_i(z_i-r_i/s_r)^2"])
    p("Giá tuyệt đối của các cổ phiếu khác nhau có thang đo khác nhau. Log return biểu diễn thay đổi tương đối và thuận tiện để dùng một mô hình chung cho nhiều mã. Target được chia cho độ lệch chuẩn return của train để giữ thang loss phù hợp; không trừ trung bình target.")
    table(["Chỉ số","Ý nghĩa"],[["MAE","Trung bình độ lớn sai số return"],["RMSE","Căn trung bình bình phương sai số; nhạy với sai số lớn"],["Directional accuracy","Tỷ lệ dấu dự báo trùng dấu return thật"],["Price MAE/RMSE","Sai số sau quy đổi về giá; nhạy thang giá"]],[150,333],caption="Các chỉ số hồi quy và giới hạn đọc kết quả.")
    p("Return đúng bằng không được xem là trạng thái riêng trong phép so dấu. Đối chứng zero-return vì thế không tự có directional accuracy 50%. Chỉ số chính để so với đối chứng là sai số return; accuracy theo hướng chỉ bổ sung thông tin.")
    p("Các biến động rất lớn vẫn được giữ trong tập đánh giá và thống kê riêng. Loại chúng chỉ vì mô hình dự báo kém sẽ làm thay đổi bài toán sau khi đã quan sát kết quả.")

    page("2.10. Lan truyền ngược qua thời gian")
    p("Backpropagation through time (BPTT) áp dụng quy tắc dây chuyền lên đồ thị đã trải. Tại bước cuối, gradient của loss theo trạng thái được truyền ngược qua tanh và qua ma trận hồi tiếp. Vì tham số dùng chung, gradient tham số cộng đóng góp từ mọi bước.")
    math([r"\delta_t=\frac{\partial L}{\partial h_t}\odot(1-h_t^2)",
          r"\nabla_{W_x}L=\sum_t x_t^\top\delta_t,\quad \nabla_{W_h}L=\sum_t h_{t-1}^\top\delta_t"])
    code("""dh = states[-1] - target
    for t in range(len(x)-1, -1, -1):
        dz = dh * (1 - states[t+1]**2)
        dwx += np.outer(x[t], dz)
        dwh += np.outer(states[t], dz)
        db += dz
        dh = dz @ wh.T
    ""","BPTT cho loss bình phương tại trạng thái cuối của ví dụ NumPy.")
    p("Đoạn minh họa dùng loss ở trạng thái cuối để thấy rõ đường đạo hàm; huấn luyện chính có thêm đầu tuyến tính và loss phân loại hoặc hồi quy. Autograd của PyTorch và GradientTape của TensorFlow tính đạo hàm cho đồ thị đầy đủ.")
    p("Cửa sổ hữu hạn giới hạn đường BPTT ở 20 hoặc 30 bước trong thiết kế này. Bài không truyền trạng thái giữa các cửa sổ, nên không phải thuật toán học trực tuyến duy trì bộ nhớ xuyên toàn bộ lịch sử.")

    page("2.11. Tiêu biến, bùng nổ và clipping")
    p("Gradient đi qua nhiều phép nhân ma trận và đạo hàm kích hoạt. Nếu độ lớn bị co lặp lại, tác động của quan sát xa có thể trở nên rất nhỏ; nếu bị khuếch đại, cập nhật có thể mất ổn định. Hai hiện tượng này là khó khăn đặc trưng của RNN [2].")
    math(r"g_{\mathrm{clip}}=g\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right)")
    code("""torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    # Equivalent global-norm clipping in TensorFlow:
    clipped, norm = tf.clip_by_global_norm(gradients, 1.0)
    ""","Giới hạn chuẩn toàn cục trước cập nhật Adam.")
    p("Với ví dụ vô hướng, 0,5 mũ 20 xấp xỉ 9,54×10⁻⁷, còn 1,5 mũ 20 xấp xỉ 3.325. Ví dụ chỉ minh họa phép nhân lặp; gradient RNN thực tế phụ thuộc trạng thái, hướng vector và các ma trận, không được xác định bằng một hệ số duy nhất.")
    p("Clipping hạn chế bước cập nhật khi gradient quá lớn, nhưng không khôi phục gradient đã tiêu biến. Khởi tạo ma trận hồi tiếp trực giao và cửa sổ vừa phải hỗ trợ ổn định ở giai đoạn đầu; chúng không chứng minh mô hình nhớ được mọi phụ thuộc dài hạn.")

    page("2.12. Kiểm tra đạo hàm bằng sai phân hữu hạn")
    math(r"\frac{\partial L}{\partial\theta_j}\approx\frac{L(\theta+\varepsilon e_j)-L(\theta-\varepsilon e_j)}{2\varepsilon}")
    p("Sai phân trung tâm đánh giá độ thay đổi của loss khi một tham số được tăng và giảm một lượng nhỏ. Phương pháp chậm khi có nhiều tham số nhưng hữu ích cho một mạng NumPy nhỏ, nơi có thể đối chiếu từng phần tử của gradient.")
    code("""numeric = finite_difference(
        lambda w: last_state_gradient(x, w, wh, bias, target)[0], wx)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-6)
    ""","Đối chiếu gradient giải tích và gradient số.")
    p("Epsilon quá lớn tạo sai số xấp xỉ; quá nhỏ có thể làm sai số làm tròn chiếm ưu thế. Ví dụ dùng float64 và epsilon 10⁻⁶, còn thực nghiệm framework dùng float32. Kết quả kiểm tra nhỏ xác nhận một đường tính toán cụ thể, không thay thế kiểm chứng dữ liệu và đánh giá mô hình.")
    table(["Kiểm tra","Đối tượng"],[["Gradient NumPy","W_x, W_h và b_h"],["Đối chiếu forward","Trạng thái cuối và đầu tuyến tính"],["Masking","Giá trị padding không đổi đầu ra"],["Checkpoint","Trạng thái optimizer và trọng số khi tiếp tục"]],[180,303],caption="Các tầng kiểm tra độc lập.")

    page("2.13. Số tham số và chi phí tính toán")
    math(r"P=DH+H^2+H+HC+C")
    table(["Dataset","D","H","C","Tham số học"],[["Retailrocket",10,32,3,1475],["S&P 500",7,32,1,1313]],[140,65,65,65,148],caption="Số tham số của kiến trúc RNN sử dụng.")
    p("PyTorch RNN cung cấp hai vector bias; Keras SimpleRNN dùng một vector. Bản PyTorch cố định bias hồi tiếp bằng không, chỉ học bias đầu vào. Nhờ vậy hai mô hình có cùng số tham số học và cùng dạng hàm; vẫn có một vector không học trong cấu trúc đối tượng PyTorch.")
    p("Số tham số nhỏ không đồng nghĩa chi phí huấn luyện không đáng kể. Hàng trăm nghìn cửa sổ phải đi qua nhiều bước hồi tiếp ở mỗi epoch. Tạo cửa sổ khi lấy batch giảm bộ nhớ lưu trữ, nhưng không loại bỏ phép tính cần thiết cho từng chuỗi.")
    p("Chi phí một forward có bậc xấp xỉ B×T×(DH+H²+HC). Biểu thức chỉ mô tả số phép tính; thời gian còn phụ thuộc framework, masking, truyền dữ liệu và kernel phần cứng.")

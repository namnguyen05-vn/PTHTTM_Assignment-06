"""Academic scope, audited datasets and implemented methodology."""
from report_engine import *
GH="https://github.com/namnguyen05-vn/PTHTTM_Assignment-06"

def front(draft):
    page("TÓM TẮT VÀ PHẠM VI",chapter=0)
    p("Báo cáo nghiên cứu mạng nơ-ron hồi tiếp cơ bản (RNN) dưới góc nhìn hàm và chuỗi thời gian. Phép truy hồi, trạng thái ẩn, chia sẻ trọng số, lan truyền ngược qua thời gian và xử lý chuỗi có độ dài khác nhau được giải thích bằng công thức, ví dụ NumPy và hai cách triển khai PyTorch/Keras.")
    p("Hai bộ dữ liệu gồm Retailrocket với 2.756.101 sự kiện hành vi và S&P 500 với 619.040 bản ghi ngày của 505 mã cổ phiếu. Nhiệm vụ tương ứng là dự đoán loại hành vi kế tiếp trong phiên và dự đoán log return của phiên giao dịch quan sát tiếp theo. Tập chia theo thời gian và chuẩn hóa dựa trên train tạo cơ sở đánh giá theo hướng quá khứ đến tương lai.")
    if draft:
        p("<b>Trạng thái tài liệu:</b> bản thảo trước thực nghiệm chính. Các số liệu dữ liệu và kiểm chứng chương trình đã được đo; kết quả chạy thử nhỏ không được trình bày như chất lượng dự đoán trên test. Phần kết quả và kết luận thực nghiệm chỉ được xuất khi có đủ bốn lượt huấn luyện chính.",small=True)
    else:
        p("Bốn lượt RNN được kiểm tra từ tệp dự đoán, đối chiếu với các phương pháp cơ sở và trực quan hóa theo nhiệm vụ. Việc so sánh hai framework dựa trên cùng dữ liệu, độ dài cửa sổ, số chiều ẩn và khởi tạo; không giả định số học giữa hai backend hoàn toàn giống nhau.")
    p("<b>Từ khóa:</b> RNN; trạng thái ẩn; BPTT; hành vi khách hàng; chuỗi thời gian; dự báo; PyTorch; Keras.",small=True)
    p("Nguồn và tài nguyên: "+link(GH)+". Trợ lý AI hỗ trợ triển khai, kiểm tra và biên soạn; các thống kê dữ liệu và kết quả kiểm tra được sinh từ tệp thực tế.",small=True)
    page("MỤC LỤC (1/2)",chapter=0)
    page("MỤC LỤC (2/2)",chapter=0)

def introduction():
    page("CHƯƠNG 1. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU",chapter=1)
    p("Nhiều hệ thống thông minh nhận dữ liệu không chỉ khác nhau về giá trị mà còn khác nhau về thứ tự xuất hiện. Một quyết định dự đoán cần xác định thông tin nào đã có tại thời điểm ra quyết định. Vì vậy, bài toán chuỗi yêu cầu đồng thời mô hình hóa phụ thuộc và tổ chức quy trình đánh giá phù hợp thời gian.")
    table(["Câu hỏi nghiên cứu","Bằng chứng"],[["RNN biểu diễn lịch sử bằng hàm như thế nào?","Phương trình truy hồi, ví dụ số và BPTT"],["Hai dataset có cấu trúc chuỗi gì?","Audit, phân bố, ranh giới thực thể và thời gian"],["PyTorch và Keras triển khai cùng ý tưởng ra sao?","Đồ thị, masking và đối chiếu NumPy"],["RNN có cải thiện so với dự đoán đơn giản?","Test theo thời gian và các đối chứng"],["Kết quả có thể kiểm tra và tiếp tục không?","Checkpoint, optimizer, dự đoán và hồ sơ kiểm chứng"]],[260,223],caption="Các câu hỏi và cách khảo sát.")
    p("Phạm vi chỉ gồm RNN cơ bản với một lớp hồi tiếp tanh và một đầu tuyến tính. Hai framework là hai cách cài đặt cùng họ mô hình, không phải hai thuật toán học khác nhau. Không đặt mục tiêu tìm cấu hình tối ưu toàn cục hoặc đưa ra chiến lược giao dịch.")

    page("1.2. Đơn vị dự đoán và giới hạn quan sát")
    p("Với Retailrocket, một mẫu là lịch sử từ một đến hai mươi sự kiện đã quan sát trong cùng phiên và loại sự kiện kế tiếp. Với chứng khoán, một mẫu là ba mươi phiên quan sát trước đó của cùng mã và log return của phiên tiếp theo.")
    table(["Yếu tố","Retailrocket","S&P 500"],[["Thực thể","Khách truy cập và phiên","Mã cổ phiếu"],["Trục thứ tự","Timestamp sự kiện","Ngày giao dịch"],["Đầu vào","Hành vi, khoảng cách, lịch thời gian","Đặc trưng giá/khối lượng quá khứ"],["Đầu ra","View/cart/transaction","Log return"],["Giới hạn","Có điều kiện phiên còn sự kiện","Dữ liệu lịch sử và nguồn giá"]],[100,191,192],caption="Đơn vị phân tích ở hai nhiệm vụ.")
    p("Hành vi giao dịch không đồng nhất với số đơn hàng hoặc doanh thu; nhiều dòng có thể thuộc cùng một giao dịch. Dự đoán loại hành vi tiếp theo cũng không tương đương dự đoán khả năng mua của mọi khách truy cập, vì khách không còn sự kiện không có nhãn kế tiếp trong thiết kế.")
    p("Đối với chứng khoán, dữ liệu là một panel nhiều chuỗi. Trộn ngẫu nhiên các cửa sổ vào train và test sẽ khiến quan sát tương lai có thể tham gia học trước khi đánh giá quá khứ. Bài giữ ranh giới thời gian toàn cục chung cho mọi mã.")

def datasets():
    retail=read("results/retailrocket_audit.json");stock=read("results/sp500_audit.json")
    page("CHƯƠNG 3. NGUỒN VÀ KIỂM TRA DỮ LIỆU",chapter=3)
    table(["Dataset","Quy mô tệp gốc","Nguồn"],[["Retailrocket",f'{retail["raw_rows"]:,} sự kiện',"events.csv"],["S&P 500",f'{stock["raw_rows"]:,} dòng',"all_stocks_5yr.csv"]],[140,180,163],caption="Các tệp được tải và kiểm tra trực tiếp.")
    p("Retailrocket: "+link("https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset"),small=True)
    p("S&P 500: "+link("https://www.kaggle.com/datasets/camnugent/sandp500"),small=True)
    p("Trình tải lấy riêng hai tệp cần thiết, lưu SHA256 và kiểm tra tên cột. Các tệp thuộc tính sản phẩm của Retailrocket không được sử dụng vì nhiệm vụ hiện tại dự đoán loại hành vi từ lịch sử tương tác. Quy mô được báo cáo theo tệp thật, không cộng các tệp nguồn không tham gia mô hình.")
    p("Dữ liệu raw và các mảng prepared nằm ngoài repository mã nguồn. Các mảng được truy cập bằng memory mapping, giúp tạo batch từ cửa sổ cần thiết mà không lưu trước toàn bộ tensor ba chiều.")
    p("Retailrocket công bố giấy phép CC BY-NC-SA 4.0; nguồn S&P 500 ghi CC0. Báo cáo dẫn nguồn dataset; repository cung cấp mã tải và thống kê, không nhân bản toàn bộ dữ liệu gốc.")

    page("3.2. Retailrocket: các trường và ngữ nghĩa")
    table(["Trường","Cách hiểu trong nghiên cứu"],[["timestamp","Thời điểm Unix tính bằng mili giây"],["visitorid","Định danh khách truy cập đã ẩn danh"],["event","view, addtocart hoặc transaction"],["itemid","Định danh sản phẩm; dùng kiểm tra lặp sản phẩm"],["transactionid","Mã giao dịch, không dùng làm đầu vào"]],[135,348],caption="Từ điển dữ liệu hành vi.")
    p("Timestamp được đổi về UTC để quy tắc xử lý nhất quán. Nguồn không xác định múi giờ địa phương của từng người dùng; các đặc trưng giờ và ngày trong tuần vì vậy mô tả nhịp thời gian theo UTC, không được diễn giải trực tiếp thành giờ sinh hoạt địa phương.")
    p("Thiếu transactionid ở sự kiện xem hoặc thêm giỏ hàng là phù hợp ngữ nghĩa, không tự động là lỗi cần điền. Dùng sự hiện diện của mã giao dịch để dự báo loại sự kiện cùng dòng sẽ gây lộ nhãn; bài loại cột này khỏi đặc trưng.")
    p("Itemid chỉ dùng để xác định sản phẩm ở bước hiện tại có giống bước trước trong phiên hay không. Mô hình không học embedding cho hàng trăm nghìn mã sản phẩm; lựa chọn này tập trung nghiên cứu vào động lực loại hành vi và giữ kích thước RNN nhỏ.")

    page("3.3. Retailrocket: làm sạch và dựng phiên")
    table(["Bước kiểm tra","Số lượng"],[["Dòng gốc",retail["raw_rows"]],["Trùng hoàn toàn bị loại",retail["exact_duplicates_removed"]],["Dòng trùng thời điểm gây thứ tự mơ hồ bị loại",retail["ambiguous_timestamp_rows_removed"]],["Sự kiện còn lại",retail["retained_events"]],["Phiên sau xử lý",retail["sessions"]],["Phiên một sự kiện",retail["singleton_sessions"]]],[355,128],caption="Audit làm sạch Retailrocket.")
    p("Sau khi loại bản ghi trùng hoàn toàn, các dòng cùng visitorid và timestamp nhưng còn nhiều quan sát được loại khỏi chuỗi. Đây là quy tắc bảo thủ để không tự đặt thứ tự giữa các hành động đồng thời. Nó làm mất một phần sự kiện giao dịch nhiều sản phẩm; phân bố sau xử lý không được đồng nhất với toàn bộ lưu lượng ban đầu.")
    p("Một phiên mới bắt đầu khi đổi visitorid hoặc khoảng cách từ sự kiện trước vượt ba mươi phút. Ngưỡng này là giả định mô hình hóa của nghiên cứu, không phải nhãn phiên do website cung cấp. Phiên chỉ có một sự kiện không tạo được cặp lịch sử–hành vi tiếp theo.")
    p("Các phiên cắt qua ranh giới tập chia bị loại khỏi nhóm mẫu huấn luyện/đánh giá để một phiên không xuất hiện ở nhiều phần. Cùng một khách quay lại ở thời gian khác vẫn có thể xuất hiện ở nhiều tập; bài đánh giá dự báo tương lai của dòng truy cập, không chỉ riêng khách hoàn toàn mới.")

    page("3.4. Retailrocket: đặc trưng và mất cân bằng")
    fig("retail_distribution.png","Phân bố nhãn mục tiêu và độ dài lịch sử sau xử lý.",maxheight=205)
    table(["Nhóm đặc trưng","Biểu diễn"],[["Hành vi","Ba biến one-hot"],["Khoảng cách","log(1 + số giây từ sự kiện trước)"],["Lặp sản phẩm","Cờ cùng itemid với bước trước"],["Vị trí trong phiên","log(1 + chỉ số vị trí)"],["Nhịp thời gian","Sin/cos của giờ và ngày trong tuần"]],[155,328],caption="Mười đặc trưng của một bước hành vi.")
    p("Các biến đều được tính từ sự kiện hiện tại và quá khứ trong cửa sổ. Không dùng thời lượng cuối cùng của phiên, tổng số giao dịch trong tương lai hoặc số bước còn lại làm đặc trưng. Những biến đó chỉ biết sau thời điểm dự báo và có thể tạo kết quả lạc quan giả.")
    p("Nhãn view áp đảo nên accuracy cần được đọc cùng macro-F1 và chỉ số từng lớp. Histogram độ dài có mức chặn ở hai mươi bước: giá trị 20 gồm cả lịch sử dài hơn đã được cắt lấy phần gần nhất.",small=True)

    page("3.5. Retailrocket: nhịp thời gian và chuyển hành vi")
    fig("retail_daily.png","Số nhãn mục tiêu theo ngày; đây là các mẫu dự đoán hợp lệ, không phải mọi sự kiện raw.",maxheight=195)
    fig("retail_transition.png","Xác suất chuyển hành vi ước lượng chỉ từ train, chưa cộng làm trơn.",maxheight=230)
    p("Ma trận chuyển bậc một chỉ dùng hành vi gần nhất. Nó là một đối chứng cần thiết: nếu RNN không vượt phương pháp này, chưa có bằng chứng rằng lịch sử dài hơn và các đặc trưng bổ sung mang lại lợi ích trong cấu hình đã khảo sát.",small=True)

    page("3.6. S&P 500: cấu trúc và kiểm tra giá")
    table(["Trường","Ý nghĩa"],[["date","Ngày quan sát"],["Name","Mã cổ phiếu"],["open / close","Giá mở / đóng cửa"],["high / low","Giá cao / thấp nhất ngày"],["volume","Khối lượng giao dịch"]],[145,338],caption="Từ điển dữ liệu giá và khối lượng.")
    table(["Kiểm tra","Số lượng"],[["Dòng gốc",stock["raw_rows"]],["Dòng không hợp lệ bị loại",stock["invalid_rows_removed"]],["Xung đột mã/ngày bị loại",stock["conflicting_date_rows_removed"]],["Dòng còn lại",stock["retained_rows"]],["Mã cổ phiếu",stock["ticker_count"]]],[330,153],caption="Audit dữ liệu S&P 500.")
    p("Giá phải hữu hạn và dương, khối lượng không âm; high và low phải bao quanh open và close. Các đoạn có khoảng cách hơn mười ngày lịch được tách để cửa sổ không nối qua khoảng thiếu dài. Một vài ngày nghỉ không được điền thành giao dịch giả.")
    p("Thẻ Kaggle không xác lập đầy đủ chính sách điều chỉnh corporate action theo thông tin có sẵn từng thời điểm. Bài ghi nhận giới hạn này, cùng nguy cơ thiên lệch do thành phần chỉ số và các mã còn hiện diện trong bộ dữ liệu.")

    page("3.7. S&P 500: biến đổi nhân quả và cửa sổ")
    math([r"r_t=\log(C_t/C_{t-1}),\quad q_t=\log(C_t/O_t)",
          r"a_t=\log(H_t/L_t),\quad g_t=\log(O_t/C_{t-1})"])
    table(["Đặc trưng","Thông tin sử dụng"],[["Log return đóng cửa","Close hiện tại và close trước"],["Return trong ngày","Close / open"],["Biên độ trong ngày","High / low"],["Gap mở cửa","Open hiện tại / close trước"],["Đổi log volume","Chênh lệch log(1 + volume)"],["Ngày trong tuần","Hai thành phần sin/cos"]],[185,298],caption="Bảy đặc trưng của một bước chứng khoán.")
    p("Các đại lượng trong ngày t được dùng để dự báo phiên t+1; high, low hoặc volume của t+1 không xuất hiện trong đầu vào. Cửa sổ chỉ thuộc một mã cổ phiếu. Một dòng đầu đoạn cần giá trước để tạo đặc trưng nên không được dùng như bước đủ thông tin.")
    p("Khung thời gian là ba mươi phiên quan sát liên tiếp, không phải ba mươi ngày lịch. Target là return từ giá cuối cửa sổ đến giá ở quan sát kế tiếp còn hợp lệ. Tên gọi một phiên quan sát tiếp theo giữ rõ giới hạn khi dữ liệu có ngày thiếu.")

    page("3.8. S&P 500: phân bố và độ phủ")
    fig("stock_exploration.png","Phân bố log return và ba chuỗi giá chuẩn hóa được chọn trước để minh họa.",maxheight=215)
    fig("stock_windows.png","Số cửa sổ hợp lệ theo từng mã cổ phiếu.",maxheight=165)
    p(f'Có {stock["extreme_target_returns_above_50pct"]} target với độ lớn log return vượt log(1,5). Ngưỡng được dùng để thống kê, không loại nhãn đánh giá. Sai số lớn có thể liên quan biến động thị trường, corporate action hoặc chất lượng nguồn; bộ dữ liệu hiện tại không đủ xác nhận từng nguyên nhân.',small=True)
    p("Histogram return giới hạn khung hiển thị ở ±0,1 để thấy phần tập trung; đó không phải cắt ngọn dữ liệu khi huấn luyện hoặc tính metric. Mật độ được chuẩn hóa trong khung hiển thị. Đồ thị giá chuẩn hóa giúp so hình dạng nhưng không thay thế đánh giá sai số return.",small=True)

    page("3.9. Tập chia và thống kê chuẩn hóa")
    rows=[]
    for name,m in [("Retailrocket",retail),("S&P 500",stock)]:
        rows.append([name]+[m["split_sizes"][k] for k in ["train","val","test"]])
    table(["Dataset","Train","Validation","Test"],rows,[135,116,116,116],caption="Số mẫu dự đoán sau tạo cửa sổ.")
    for name,m in [("Retailrocket",retail),("S&P 500",stock)]:
        p(name+": ranh giới validation bắt đầu "+m["cutoffs_utc"][0][:10]+", test bắt đầu "+m["cutoffs_utc"][1][:10]+".",small=True)
    math(r"x'_j=(x_j-\mu_{j,\mathrm{train}})/\sigma_{j,\mathrm{train}}")
    p("Ranh giới lấy theo khoảng 70% và 85% các ngày quan sát khác nhau. Tỷ lệ số dòng vì vậy không nhất thiết là 70/15/15. Nhãn được phân tập theo thời điểm mục tiêu, không theo vị trí tùy ý trong CSV.")
    p("Chứng khoán cho phép cửa sổ validation/test chứa lịch sử trước ranh giới vì lịch sử đó đã có tại thời điểm dự báo. Điều bị cấm là target tương lai tham gia học hoặc thống kê chuẩn hóa. Retailrocket bổ sung quy tắc không chia đôi một phiên.")
    p("Chuẩn hóa được fit trên các hàng đặc trưng ở giai đoạn train, trước khi lấy các cửa sổ lặp chồng lên nhau. Nhờ vậy một dòng raw không bị nhân trọng số chỉ vì xuất hiện trong nhiều cửa sổ. Biến có độ lệch chuẩn gần không dùng thang một để tránh chia cho không.")

def implementation():
    page("CHƯƠNG 4. KIẾN TRÚC VÀ HAI FRAMEWORK",chapter=4)
    table(["Thành phần","Retailrocket","S&P 500"],[["Đầu vào","B × 20 × 10","B × 30 × 7"],["Độ dài thật","1 đến 20","30"],["RNN","32 chiều, tanh","32 chiều, tanh"],["Đầu tuyến tính","32 → 3","32 → 1"],["Tham số học","1.475","1.313"]],[140,171,172],caption="Kiến trúc chung cho hai cách triển khai.")
    p("Mô hình chỉ có một lớp RNN; không thêm cơ chế cổng hoặc mạng hai chiều. Mạng hai chiều xử lý cả hai hướng của một cửa sổ đã biết nhưng tạo thêm biến thiết kế; nghiên cứu hiện tại giữ đường truy hồi từ quá khứ đến hiện tại để giải thích và đối chiếu rõ ràng.")
    p("Hàm xây dựng được tách khỏi bộ đọc dữ liệu và bộ huấn luyện. Hai backend nhận cùng mảng float32, cùng độ dài thật và cùng nhãn. Dataset được chuẩn bị một lần, không chia lại ngẫu nhiên ở từng notebook.")
    p("Khởi tạo chung được sinh bằng NumPy: ma trận đầu vào và đầu ra dùng phân bố đều theo kích thước, ma trận hồi tiếp trực giao từ QR, bias bằng không. Seed không tự bảo đảm hai framework tạo cùng trọng số nếu mỗi framework dùng bộ sinh riêng; vì vậy bài truyền trực tiếp các ma trận chung.")

    page("4.2. RNN trong PyTorch")
    code("""rnn = torch.nn.RNN(input_dim, 32,
                       batch_first=True, nonlinearity="tanh")
    rnn.bias_hh_l0.requires_grad_(False)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    sequence_output, hidden = rnn(packed)
    logits = head(hidden[-1])
    ""","Đồ thị PyTorch và đóng gói chuỗi có độ dài khác nhau.")
    p("batch_first=True giữ tensor theo thứ tự B×T×D. Packed sequence bỏ các bước ngoài lengths; hidden[-1] là trạng thái cuối của lớp RNN duy nhất. enforce_sorted=False cho phép đầu vào không sắp giảm dần độ dài và framework khôi phục thứ tự mẫu.")
    p("PyTorch lưu ma trận đầu vào theo H×D và ma trận hồi tiếp theo H×H với quy ước nhân chuyển vị. Khi gán trọng số NumPy ở dạng vector hàng, W_x và W_h phải chuyển vị. Thiếu chuyển vị có thể gây lỗi hình dạng, hoặc tạo mô hình khác nếu các chiều tình cờ bằng nhau.")
    p("Trong adapter huấn luyện, bias_hh được gán bằng không rồi cố định; bias_ih được học. Đầu tuyến tính tạo logits hoặc một giá trị hồi quy, không gắn sẵn softmax để hàm loss nhận logits ổn định số.")

    page("4.3. Một bước cập nhật PyTorch")
    code("""optimizer.zero_grad(set_to_none=True)
    output = model(x, lengths)
    losses = torch.nn.functional.cross_entropy(
        output, labels, reduction="none")
    loss = (losses * class_weight[labels]).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
    optimizer.step()
    ""","Bước học phân loại; đầu hồi quy thay loss bằng MSE.")
    p("zero_grad xóa gradient tích lũy từ batch trước. backward áp dụng quy tắc đạo hàm trên đồ thị forward hiện tại. Gradient clipping được thực hiện sau backward và trước optimizer.step, nên giới hạn vector gradient dùng cho bước cập nhật.")
    p("Adam duy trì moment bậc một và bậc hai cùng số bước. Chỉ lưu state_dict của mô hình đủ để suy luận nhưng không đủ để tiếp tục Adam đúng trạng thái. Checkpoint của nghiên cứu lưu cả model, optimizer và trạng thái bộ sinh ngẫu nhiên PyTorch.")
    p("Loss epoch là tổng loss batch nhân số mẫu rồi chia tổng mẫu. Batch cuối nhỏ không được nhận trọng số ngang một batch đầy đủ. Bài không dùng gradient accumulation hoặc mixed precision để giữ quy trình số học dễ đối chiếu.")

    page("4.4. SimpleRNN trong Keras")
    code("""sequence = tf.keras.Input((length, input_dim))
    sizes = tf.keras.Input((), dtype="int32")
    mask = tf.sequence_mask(sizes, maxlen=length)
    hidden = tf.keras.layers.SimpleRNN(
        32, activation="tanh")(sequence, mask=mask)
    output = tf.keras.layers.Dense(output_dim)(hidden)
    model = tf.keras.Model([sequence, sizes], output)
    ""","Keras Functional API với lengths làm đầu vào tường minh.")
    p("SimpleRNN nhận hai ma trận kernel và recurrent_kernel cùng một vector bias. Với return_sequences=False mặc định, lớp trả trạng thái cuối hợp lệ sau khi xét mask. Mặt nạ được xây dựng theo số bước thật, không dựa trên giá trị của đặc trưng.")
    p("Functional API biểu diễn rõ hai đầu vào: dữ liệu và lengths. Trong phiên bản TensorFlow/Keras dùng cho thực nghiệm, tf.sequence_mask tạo thao tác đồ thị tương thích với tensor Keras. Mã và phiên bản được lưu cùng hồ sơ môi trường.")
    p("Bản triển khai dùng tf.keras đi kèm TensorFlow 2.10.1 trong môi trường Anaconda đã có trên máy. Nó không ngầm chuyển sang Keras 3 hay backend khác trong các lượt so sánh.")

    page("4.5. GradientTape và checkpoint Keras")
    code("""with tf.GradientTape() as tape:
        logits = model([x, lengths], training=True)
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, 3), logits=logits)
        loss = tf.reduce_mean(losses * tf.gather(weights, labels))
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, norm = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ""","Bước học Keras theo cùng mẫu số và trọng số lớp.")
    p("GradientTape ghi các phép biến đổi cần thiết để lấy đạo hàm theo biến học. tf.function biên dịch bước học để giảm chi phí gọi Python; lần gọi đầu có chi phí tạo đồ thị nên thời gian chạy thử nhỏ không thể ngoại suy đơn giản sang toàn bộ tập.")
    p("tf.train.Checkpoint lưu mô hình và optimizer, bao gồm các biến moment đã được tạo. Con trỏ state.json chỉ được cập nhật sau khi checkpoint hoàn tất. Nếu ngắt khi đang ghi, lần chạy sau vẫn đọc epoch hoàn chỉnh trước đó.")
    p("Bài không dùng dropout trong mô hình thực nghiệm. Thứ tự batch được sinh lại từ seed cộng số epoch. Thiết kế này giảm trạng thái ngẫu nhiên phải theo dõi khi khởi động lại; kết quả tái lập vẫn phụ thuộc điều kiện backend và thiết bị.")

    page("4.6. Đối chiếu cài đặt và giới hạn tương đương")
    checks=[read("results/verification_"+f+".json") for f in ["pytorch","keras"]]
    table(["Framework","Sai số forward lớn nhất","Sai số bước tiếp tục"],[[r["framework"],f'{r["initial_numpy_max_error"]:.2e}',f'{r["optimizer_resume_max_error"]:.2e}'] for r in checks],[130,190,163],caption="Kiểm tra CPU trên ví dụ nhỏ với cùng trọng số.")
    p("Hai cách triển khai có cùng hàm toán học khi trọng số được ánh xạ đúng. Sai số nhỏ còn lại xuất phát từ số thực hữu hạn và thứ tự phép tính. Tương đương forward không đồng nghĩa mọi cập nhật Adam và mọi kernel GPU khớp bitwise.")
    p("Cùng epsilon và learning rate chưa loại bỏ mọi khác biệt triển khai optimizer. Vì vậy báo cáo đối chiếu chất lượng, đường học và chi phí của hai framework trên cùng thiết kế, không coi chênh lệch nhỏ là bằng chứng một framework có khả năng biểu diễn tốt hơn.")
    table(["Giữ chung","Có thể khác"],[["Dữ liệu và ranh giới thời gian","Kernel và thứ tự cộng số thực"],["Trọng số ban đầu từ NumPy","Chi tiết thực thi optimizer"],["Loss, trọng số lớp, clipping","Chi phí packed sequence/masking"],["Cửa sổ, hidden size, seed","Thời gian khởi tạo đồ thị"]],[240,243],caption="Điều kiện so sánh và các nguồn khác biệt còn lại.")

def methods():
    page("CHƯƠNG 5. THIẾT KẾ THỰC NGHIỆM",chapter=5)
    config=read("configs/main.json")
    table(["Thành phần","Thiết lập"],[["Dataset","Retailrocket; S&P 500"],["Framework","PyTorch; Keras/TensorFlow"],["Mô hình","Simple RNN một lớp, tanh, H=32"],["Số lượt chính","2 × 2 = 4"],["Seed","42"],["Batch size",config["batch_size"]],["Epoch tối đa",config["max_epochs"]],["Learning rate",config["learning_rate"]],["Clipping norm",config["clip_norm"]],["Early stopping","Patience 4; min_delta 0,00001"]],[180,303],caption="Các điều kiện chính của thực nghiệm.")
    p("Một seed được dùng cho bốn cấu hình chính để tập trung vào RNN và kiểm soát thời gian chạy. Không báo cáo trung bình hay độ lệch chuẩn giữa các seed khi chỉ có một lượt cho mỗi tổ hợp. Kết quả phản ánh cấu hình cụ thể, chưa định lượng đầy đủ biến thiên khởi tạo.")
    p("Mỗi lượt chạy trong tiến trình riêng để tránh giữ đồng thời hai framework trên GPU. Notebook minh họa chạy CPU và mặc định không khởi động huấn luyện đầy đủ. Hai loại hoạt động tạo đầu ra khác nhau và được lưu ở thư mục riêng.")

    page("5.2. Validation, chọn checkpoint và test")
    table(["Nhiệm vụ","Tiêu chí chọn","Metric bổ sung"],[["Retailrocket","Macro-F1 validation lớn hơn","Cross-entropy, accuracy, từng lớp, AP"],["S&P 500","RMSE return validation nhỏ hơn","MAE, hướng thay đổi, sai số giá"]],[140,170,173],caption="Quy tắc chọn mô hình trước khi đánh giá test.")
    p("Một checkpoint mới được chọn khi tiêu chí cải thiện quá min_delta so với checkpoint đang giữ. Nếu không cải thiện qua bốn epoch liên tiếp, lượt dừng sớm. Sau khi kết thúc ngân sách hoặc dừng sớm, checkpoint được chọn được tải lại rồi mới sinh dự đoán test.")
    p("Loss train của Retailrocket có trọng số lớp, còn cross-entropy validation không có trọng số. Hai đường vì vậy phản ánh các mục tiêu khác nhau; khoảng cách giữa chúng không chỉ đo quá khớp. Với chứng khoán, loss train dùng target đã chia thang, còn RMSE validation giữ đơn vị log return gốc.")
    p("Test không dùng để chọn epoch, trọng số lớp, scaler hoặc siêu tham số. Những phân tích sau test mang tính mô tả; nếu được dùng để cải tiến mô hình, vòng đánh giá sau cần một thiết kế kiểm tra độc lập phù hợp.")

    page("5.3. Phương pháp đối chứng")
    table(["Dataset","Đối chứng","Thông tin sử dụng"],[["Retailrocket","Prior toàn train","Tần suất ba hành vi"],["Retailrocket","Markov bậc một","Hành vi gần nhất và bảng chuyển train"],["S&P 500","Zero-return","Giá dự báo bằng giá gần nhất"],["S&P 500","Mean-return","Return trung bình toàn train"]],[125,140,218],caption="Các mốc so sánh đơn giản.")
    p("Prior và Markov dùng làm trơn cộng một trước khi chuyển thành xác suất, tránh log của không. Markov kiểm tra liệu trạng thái ẩn RNN có mang lại lợi ích so với việc chỉ nhớ hành vi ngay trước đó. Các xác suất đều được ước lượng từ train.")
    p("Zero-return là đối chứng quan trọng vì giá thường thay đổi tương đối ít giữa hai phiên liên tiếp. Mean-return thêm một mức trôi ước lượng từ train. Cả hai không được cập nhật bằng nhãn test trong quá trình đánh giá.")
    p("Đối chứng không được gọi là yếu chỉ vì ít tham số. Nếu phương pháp đơn giản đạt sai số thấp hơn RNN, đó là kết quả có ý nghĩa về giá trị của cấu hình hồi tiếp trong dữ liệu và ngân sách hiện tại.")

    page("5.4. Dừng, lưu và tiếp tục thực nghiệm")
    code("""adapter.save(checkpoint_for(epoch))
    state["epoch"] = epoch
    state["history"].append(epoch_record)
    write_json_atomic("state.json", state)
    ""","Thứ tự công bố epoch đã hoàn tất.")
    p("Checkpoint đầy đủ được lưu ở cuối epoch, sau đánh giá validation. Tệp state.json được thay thế nguyên tử để chỉ trỏ tới trạng thái đã ghi xong. Các checkpoint gần nhất và tốt nhất được giữ; bản không còn được tham chiếu được dọn nhằm hạn chế dung lượng.")
    p("Yêu cầu tạm dừng có hiệu lực ở ranh giới epoch. Nếu tiến trình bị ngắt ngay giữa epoch, phần cập nhật chưa được công bố sẽ được thực hiện lại từ đầu epoch khi chạy tiếp. Việc này khác với giữ nguyên tiến trình trong RAM khi hệ điều hành Sleep.")
    resume=read("results/resume_verification.json")
    table(["Framework","Khởi động lại tiến trình","Sai khác đầu ra lớn nhất"],[[r["framework"],"Đã kiểm tra",r["max_output_error"]] for r in resume],[130,190,163],caption="So sánh smoke hai epoch liền với dừng sau epoch thứ nhất.")
    p("Kiểm tra đạt trong môi trường hiện tại không bảo đảm kết quả giống từng bit khi đổi thiết bị, thư viện hoặc cấu hình. Các tệp config và phiên bản đi cùng checkpoint là điều kiện cần để diễn giải phép chạy tiếp.",small=True)

    page("5.5. Hồ sơ kiểm chứng và giới hạn trước huấn luyện")
    notebooks=read("results/notebook_verification.json")
    table(["Kiểm tra","Kết quả"],[["Tập chia theo thời gian","Hai dataset đạt"],["BPTT NumPy","Gradient số và giải tích khớp"],["Forward hai framework","Khớp ví dụ NumPy"],["Masking","Đổi padding không đổi dự đoán"],["Tiếp tục optimizer","Đã đối chiếu bước tiếp theo"],["Chạy lại qua tiến trình","Hai framework đạt"],["Notebook thực thi",len(notebooks)],["Cell code",sum(r["code_cells"] for r in notebooks)]],[295,188],caption="Các kiểm tra chương trình đã thực hiện.")
    p("Smoke dùng tối đa 512 mẫu train, 128 mẫu validation và hai epoch. Nó kiểm tra đường đi của chương trình trên dữ liệu thật nhưng không đo chất lượng mô hình chính. Test chính thức không được dùng làm tập đánh giá của smoke.")
    p("Sau thực nghiệm chính, dự đoán được lưu cùng ID, nhãn và thời điểm để tính lại metric. Trọng số được tải lại độc lập để đối chiếu một batch. Lịch sử, cấu hình và checkpoint giúp truy vết cách chọn mô hình thay vì chỉ lưu một bảng điểm cuối.")
    p("Nguồn code, notebook và hướng dẫn môi trường được công bố tại "+link(GH)+". Dữ liệu raw, cache và tệp tạm đặt trên ổ E, tách khỏi bộ bài nộp và lịch sử Git.")

def finish(draft):
    page("GIỚI HẠN VÀ PHẠM VI DIỄN GIẢI",chapter=7)
    p("Một tập chia theo thời gian và một seed cho mỗi cấu hình chưa đủ để kết luận về mọi giai đoạn thị trường hoặc mọi website. Khác biệt phân phối giữa giai đoạn train và test có thể ảnh hưởng chất lượng dù chương trình đúng về số học.")
    p("Retailrocket chỉ xét các phiên còn sự kiện kế tiếp; phiên một sự kiện không có nhãn trong bài toán này. Việc loại timestamp mơ hồ ảnh hưởng một phần giao dịch nhiều sản phẩm. Các chỉ số không được diễn giải thành tỷ lệ chuyển đổi của toàn bộ khách truy cập.")
    p("Bộ S&P 500 là dữ liệu lịch sử với giới hạn về thành phần chỉ số và điều chỉnh giá. Dự báo return hoặc hướng thay đổi chưa tạo thành đánh giá chiến lược giao dịch vì chưa mô hình hóa chi phí, thanh khoản và khả năng thực thi.")
    p("RNN cơ bản có cửa sổ hữu hạn, trạng thái ẩn nhỏ và khó khăn về phụ thuộc dài hạn. Nghiên cứu tập trung giải thích và kiểm chứng mô hình này; không coi việc tăng độ phức tạp là bảo đảm cải thiện.")
    if draft:
        p("Ở giai đoạn bản thảo, có thể kết luận chương trình và quy trình kiểm tra đã được chuẩn bị trên dữ liệu thật. Chưa có kết luận định lượng về ưu thế của RNN so với các đối chứng trên test chính thức.")

    refs=[
        ("[1]","Elman, J. L. Finding Structure in Time. Cognitive Science, 14, 179–211, 1990.","https://doi.org/10.1207/S15516709COG1402_1"),
        ("[2]","Pascanu, R.; Mikolov, T.; Bengio, Y. On the difficulty of training recurrent neural networks. ICML, 2013.","https://proceedings.mlr.press/v28/pascanu13.html"),
        ("[3]","PyTorch. RNN: định nghĩa toán học, hình dạng đầu vào và đầu ra.","https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html"),
        ("[4]","Keras. SimpleRNN: trạng thái, masking và phép hồi tiếp.","https://keras.io/2/api/layers/recurrent_layers/simple_rnn/"),
        ("[5]","Retailrocket. Recommender system dataset. Kaggle.","https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset"),
        ("[6]","Nugent, C. S&P 500 stock data. Kaggle.","https://www.kaggle.com/datasets/camnugent/sandp500"),
        ("[7]","Chollet, F. Deep Learning with Python, Second Edition. Manning, 2021.",""),
        ("[8]","Keras. Saving and serialization: trạng thái mô hình và optimizer.","https://keras.io/2/api/models/model_saving_apis/")]
    for part in range(2):
        page("TÀI LIỆU THAM KHẢO ("+str(part+1)+"/2)",chapter=8 if part==0 else None)
        for tag,title,url in refs[part*4:part*4+4]:
            p("<b>"+tag+"</b> "+title)
            if url:p(link(url),small=True)
        if part==1:
            p("Mã nguồn: "+link(GH),small=True)
            p("Số lượng bản ghi và các tập chia được đo trực tiếp từ tệp tải. Tài liệu API giải thích khái niệm; các phiên bản thực nghiệm cụ thể được ghi trong hồ sơ môi trường.")
    page("PHỤ LỤC. THUẬT NGỮ",chapter=9)
    table(["Thuật ngữ","Giải thích"],[
        ["RNN","Mạng cập nhật trạng thái bằng đầu vào và trạng thái trước"],
        ["Hidden state","Vector biểu diễn lịch sử đã quan sát"],
        ["Recurrence","Quy tắc truy hồi được áp dụng qua các bước"],
        ["Time step","Vị trí quan sát; không luôn là khoảng thời gian cố định"],
        ["Unrolling","Trải đồ thị hồi tiếp thành chuỗi bước tính"],
        ["BPTT","Lan truyền ngược qua đồ thị đã trải theo thời gian"],
        ["Padding / masking","Đệm tensor / bỏ qua vị trí không phải dữ liệu thật"],
        ["Logits","Điểm số trước softmax"],
        ["Log return","Logarit tỷ lệ giá liên tiếp"],
        ["Lookback window","Cửa sổ lịch sử dùng dự báo"],
        ["Data leakage","Thông tin không sẵn có đi vào học hoặc lựa chọn"],
        ["Checkpoint","Trạng thái đã lưu của mô hình và quy trình tối ưu"],
        ["Epoch","Một lượt duyệt toàn bộ tập train của cấu hình"],
        ["Early stopping","Dừng theo tiêu chí validation đã quy định"],
        ["Baseline","Phương pháp đối chứng để đánh giá giá trị bổ sung"],
        ["Concept drift","Quan hệ thống kê thay đổi theo thời gian"]],[150,333],caption="Thuật ngữ theo ngữ cảnh báo cáo.")

def indexes():
    entries=[(i+1,q["title"]) for i,q in enumerate(PAGES) if i>2]
    middle=(len(entries)+1)//2
    for dest,part in zip([PAGES[1],PAGES[2]],[entries[:middle],entries[middle:]]):
        dest["blocks"]=[Paragraph(html.escape(title)+f" <b>… {number}</b>",ST["toc"]) for number,title in part]
    (ROOT/"results/report_caption_index.json").write_text(json.dumps(dict(figures=FIGURES,tables=TABLES,code_listings=CODES),ensure_ascii=False,indent=2),encoding="utf-8")

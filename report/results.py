"""Result discussion derived from verified full experiments, never smoke scores."""
from report_engine import *
import pandas as pd

def result_pages():
    baseline=read("results/baselines.json")
    base={r["model"]:r for r in baseline}
    all_metrics={d:{f:read(f"results/runs/{d}_{f}_seed42/metrics.json")
        for f in ["pytorch","keras"]} for d in ["retailrocket","sp500"]}
    paired=read("results/paired_framework_analysis.json")
    retail=all_metrics["retailrocket"];stock=all_metrics["sp500"]
    page("CHƯƠNG 6. KẾT QUẢ VÀ THẢO LUẬN",chapter=6)
    p("Các số đo trong chương này được tính trên toàn bộ test, sau khi checkpoint được chọn bằng validation. Retailrocket có 118.192 mẫu; S&P 500 có 95.007 mẫu. Mỗi cấu hình dùng seed 42. Các lượt chạy thử nhỏ chỉ phục vụ kiểm tra chương trình và không tham gia bảng kết quả.")
    sub("6.1. Kết quả tổng thể trên Retailrocket")
    rows=[[f,pct(m["accuracy"]),dec(m["macro_f1"],6),dec(m["log_loss"],6)] for f,m in retail.items()]
    rows += [[label,pct(base[key]["accuracy"]),dec(base[key]["macro_f1"],6),dec(base[key]["log_loss"],6)] for key,label in [("prior","Tần suất train"),("markov","Markov bậc một")]]
    table(["Mô hình","Accuracy (%)","Macro-F1","Log loss"],rows,[145,105,113,120],caption="Kết quả Retailrocket. Accuracy và F1 càng cao càng tốt; log loss càng thấp càng tốt.")
    delta=retail["pytorch"]["macro_f1"]-base["markov"]["macro_f1"]
    p(f'RNN PyTorch đạt macro-F1 {dec(retail["pytorch"]["macro_f1"],4)}, cao hơn Markov {dec(delta,4)} điểm trên thang 0-1. Keras cho kết quả gần tương đương. Tuy nhiên, accuracy của cả hai RNN chỉ khoảng 90,1%, thấp hơn mức 91,56% của đối chứng luôn chọn view.')
    p("Hai đối chứng đều chọn view khi lấy xác suất lớn nhất: trong cả ba hàng của ma trận chuyển Markov, view vẫn là lớp có xác suất cao nhất. Vì vậy, accuracy và macro-F1 của chúng trùng nhau, nhưng log loss khác nhau do phân bố xác suất dự đoán khác nhau.")
    p("Markov có log loss tốt hơn RNN. Kết quả không chứng minh RNN vượt đối chứng trên mọi tiêu chí. Huấn luyện RNN dùng trọng số lớp và lựa chọn theo macro-F1; hai đối chứng dùng tần suất không gán trọng số. Sự cải thiện lớp hiếm có thể đến từ cả mục tiêu học và đặc trưng, không thể quy riêng cho bộ nhớ hồi tiếp khi chưa có thực nghiệm loại bỏ thành phần.")

    page("6.2. Đường học và lựa chọn mô hình hành vi")
    fig("retailrocket_learning.png","Loss train, log loss validation và macro-F1 validation; dấu sao chỉ epoch được chọn.",maxheight=245)
    table(["Framework","Epoch chọn","Macro-F1 val","Epoch dừng"],[[f,m["best_epoch"],dec(read(f"results/runs/retailrocket_{f}_seed42/state.json")["best_score"],6),m["completed_epochs"]] for f,m in retail.items()],[130,100,143,110],caption="Cả hai lượt chọn epoch 8 và dừng sau epoch 12.")
    p("Loss train giảm chủ yếu trong các epoch đầu rồi thay đổi chậm. Macro-F1 validation dao động thay vì tăng đều. Bốn epoch sau epoch 8 không vượt điểm tốt nhất đủ min_delta, nên early stopping dừng ở epoch 12 và khôi phục checkpoint epoch 8.")
    p("Log loss validation thấp nhất không nằm ở epoch được chọn. Điều này phù hợp thiết kế: tiêu chí lựa chọn là macro-F1, không phải cross-entropy. Dùng test để đổi sang một epoch có kết quả đẹp hơn sẽ phá vỡ vai trò đánh giá độc lập của test.")
    p("Loss train là cross-entropy có trọng số lớp; log loss validation không có trọng số. Do khác định nghĩa, không diễn giải trực tiếp khoảng cách giữa hai đường như độ chênh train-test của cùng một loss.")

    page("6.3. Precision, recall và F1 của từng hành vi")
    rows=[]
    for f,m in retail.items():
        for name,r in zip(["view","addtocart","transaction"],m["per_class"]):
            rows.append([f,name,dec(r["precision"],4),dec(r["recall"],4),dec(r["f1"],4),str(r["support"])])
    table(["Framework","Lớp","Precision","Recall","F1","Số mẫu"],rows,[84,105,76,73,72,73],caption="Đánh giá riêng ba loại hành vi trên cùng test.")
    p("Precision là tỷ lệ dự đoán đúng trong các mẫu được gán vào một lớp; recall là tỷ lệ tìm đúng trong toàn bộ mẫu thật của lớp đó. F1 là trung bình điều hòa của precision và recall. Macro-F1 lấy trung bình F1 của ba lớp với trọng số bằng nhau, nên lớp view không thể che lấp toàn bộ lỗi ở lớp hiếm.")
    p("Với PyTorch, transaction có recall khoảng 69,19% nhưng precision chỉ 36,16%. Mô hình tìm được nhiều giao dịch hơn đối chứng luôn-view, đồng thời tạo nhiều cảnh báo giao dịch sai. Addtocart có recall khoảng 27,35%, nghĩa là phần lớn hành vi thêm giỏ vẫn chưa được nhận diện đúng.")
    p("F1 của view gần 0,949, cao hơn đáng kể hai lớp còn lại. Kết quả phù hợp tính mất cân bằng của dữ liệu và cho thấy macro-F1 khoảng 0,582 không đồng nghĩa mọi lớp đều có F1 ở mức đó.")
    p("Các chỉ số đo loại sự kiện kế tiếp trong các mẫu có nhãn. Chúng không đo số đơn hàng, doanh thu hoặc xác suất mua của những khách đã rời phiên.")

    page("6.4. Ma trận nhầm lẫn và dạng lỗi")
    for f in ["pytorch","keras"]:
        fig("retailrocket_"+f+"_confusion.png","Ma trận "+f+": màu chuẩn hóa theo nhãn thật, số trong ô là số mẫu.",maxheight=216)
    p("Ở PyTorch, 5.248/7.683 mẫu addtocart bị gán thành view. Trong 4.397 dự đoán transaction, chỉ 1.590 mẫu đúng; 2.473 mẫu thực chất là view và 334 là addtocart. Hai ma trận gần giống nhau, cho thấy hai cách triển khai mắc các dạng lỗi tương tự.",small=True)

    page("6.5. Khả năng xếp hạng các lớp hiếm")
    fig("retail_pr.png","Đường precision-recall trên test; đường ngang là tỷ lệ xuất hiện lớp.",maxheight=235)
    table(["Mô hình","AP addtocart","AP transaction"],[[label,dec(m["ap_1"],6),dec(m["ap_2"],6)] for label,m in [("PyTorch",retail["pytorch"]),("Keras",retail["keras"]),("Markov",base["markov"]),("Tần suất train",base["prior"])]],[185,149,149],caption="Average precision (AP) cho bài toán mỗi lớp so với phần còn lại.")
    p("Average precision tổng hợp precision theo các mức tăng recall khi thay đổi ngưỡng điểm số. AP xem xét khả năng xếp hạng mẫu của từng lớp, khác với F1 tại quyết định argmax. Trong bài này, AP được tính bằng average_precision_score; không đồng nhất với mọi cách lấy diện tích hình thang dưới đường PR.")
    p("RNN đạt AP cao hơn Markov ở cả addtocart và transaction. Đây là bằng chứng bổ sung về khả năng ưu tiên các mẫu thuộc lớp hiếm, dù log loss tổng thể chưa tốt hơn. Hai nhận xét không mâu thuẫn vì chúng đo các khía cạnh khác nhau của dự đoán.")
    p("Đường PR được dùng để mô tả kết quả sau thực nghiệm. Không chọn ngưỡng triển khai bằng test và không báo cáo một ngưỡng tối ưu tìm trên test như kết quả đánh giá độc lập.")

    page("6.6. Kết quả dự báo log return S&P 500")
    rows=[[label,dec(m["mae"],8),dec(m["rmse"],8),pct(m["directional_accuracy"])] for label,m in [("PyTorch",stock["pytorch"]),("Keras",stock["keras"]),("Zero return",base["zero_return"]),("Mean return train",base["train_mean_return"])]]
    table(["Mô hình","MAE return","RMSE return","Đúng hướng (%)"],rows,[135,116,116,116],caption="Sai số trên 95.007 mẫu test; return dùng thang logarit gốc.")
    worse=(stock["pytorch"]["rmse"]/base["train_mean_return"]["rmse"]-1)*100
    p(f'RNN PyTorch có RMSE {dec(stock["pytorch"]["rmse"],8)}, cao hơn đối chứng trung bình train khoảng {dec(worse,2)}%. Keras gần như trùng kết quả. MAE cũng cao hơn cả hai đối chứng. Vì vậy, cấu hình RNN này chưa tạo giá trị dự báo bổ sung theo các thước đo sai số đã chọn.')
    p("Độ đúng hướng của RNN khoảng 50,17%, thấp hơn 53,09% của dự đoán luôn có return dương bằng trung bình train. Test có 53,09% mẫu tăng, 46,08% giảm và 0,83% không đổi. Do mất cân bằng nhẹ về hướng, mốc 50% không phải đối chứng duy nhất.")
    p("Directional accuracy trong mã dùng sign(dự đoán) = sign(thực). Zero return luôn dự đoán dấu 0, nên chỉ đúng ở mẫu không đổi và đạt khoảng 0,83%. Số này không có nghĩa đối chứng zero-return kém về RMSE; hai chỉ số phục vụ hai câu hỏi khác nhau.")
    p("Mỗi mẫu có trọng số như nhau trong chỉ số gộp. Các mẫu cùng mã và cùng ngày thị trường có thể phụ thuộc lẫn nhau; không coi 95.007 mẫu là các phép thử độc lập để suy ra ý nghĩa thống kê.")

    page("6.7. Diễn biến huấn luyện dự báo cổ phiếu")
    fig("sp500_learning.png","Loss train trên return đã chuẩn hóa và RMSE validation trên return gốc.",maxheight=240)
    table(["Framework","Epoch chọn","RMSE val tại epoch chọn","Epoch dừng"],[[f,m["best_epoch"],dec(-read(f"results/runs/sp500_{f}_seed42/state.json")["best_score"],8),m["completed_epochs"]] for f,m in stock.items()],[110,90,193,90],caption="Hai lượt đều chọn epoch 1 và dừng sau epoch 5.")
    p("Loss train tiếp tục giảm từ khoảng 0,986 xuống 0,886, trong khi RMSE validation tăng sau epoch 1. Mẫu hình này phù hợp với việc cải thiện độ khớp train không chuyển thành cải thiện trên giai đoạn kế tiếp. Nó có thể liên quan tới quá khớp và khác biệt phân phối; riêng các đường học chưa phân biệt được hai nguyên nhân.")
    p("Bộ huấn luyện đã khôi phục checkpoint epoch 1 trước khi dự đoán test. Kết quả không phải dự đoán của epoch 5. Early stopping được kiểm tra lại trực tiếp từ history, min_delta và patience, đồng thời checkpoint được nạp trong tiến trình mới.")
    p("Không kéo dài huấn luyện hoặc chỉnh cấu hình theo test sau khi quan sát kết quả này. Một nghiên cứu tiếp theo có thể kiểm tra cửa sổ hoặc số chiều ẩn của Simple RNN bằng nhiều giai đoạn validation, sau đó cần một khoảng test chưa được sử dụng.")

    for index,f in enumerate(["pytorch","keras"],8):
        page(f"6.{index}. Giá và return dự báo: "+f)
        fig("sp500_"+f+"_forecast.png","AAPL trên test: giá đóng cửa và log return của phiên quan sát kế tiếp.",maxheight=340)
        m=stock[f]
        p("AAPL được quy định là mã minh họa trong mã vẽ trước khi chạy thực nghiệm. Hình chỉ mô tả một mã, còn các metric chính được tính trên toàn bộ tập test nhiều mã.")
        p("Giá dự báo được tái dựng từ giá thật cuối cửa sổ nhân với exp(return dự báo). Vì giá phiên liền trước đã gần giá phiên kế tiếp, đường giá có thể bám sát thực tế ngay cả khi dự báo return ít thông tin. Panel return giúp nhận ra giới hạn này.")
        p(f'Trên toàn bộ test, MAE giá của {f} là {dec(m["price_mae"],6)} và RMSE giá là {dec(m["price_rmse"],6)}, đều cao hơn đối chứng zero-return lần lượt {dec(base["zero_return"]["price_mae"],6)} và {dec(base["zero_return"]["price_rmse"],6)}. Sai số giá gộp chịu ảnh hưởng của mức giá khác nhau giữa các mã.')
        p("Đây là dự đoán một bước có cập nhật quan sát thật ở mỗi cửa sổ. Không diễn giải hình thành dự báo nhiều bước liên tiếp hoặc bằng chứng về lợi nhuận giao dịch.",small=True)

    page("6.10. Sai số theo phân phối và thời gian")
    fig("stock_diagnostics.png","Trái: 6.000 mẫu chọn với seed 42, khung return ±0,08. Phải: histogram sai số trong khung ±0,06.",maxheight=205)
    fig("stock_monthly.png","RMSE theo tháng test. Tháng đầu và cuối chỉ bao gồm phần thời gian có trong test.",maxheight=175)
    a=paired["sp500"]
    p(f'Độ lệch chuẩn return dự đoán PyTorch là {dec(a["pytorch_return_std"],6)}, nhỏ hơn độ lệch chuẩn return thật {dec(a["actual_return_std"],6)}. Mô hình tạo tín hiệu dao động hẹp, không tái hiện tốt các biến động lớn. Các khung hình chỉ giới hạn vùng hiển thị; metric vẫn dùng toàn bộ mẫu.')
    p("Ở cả mười tháng được quan sát, RMSE của hai RNN đều cao hơn zero-return. Tháng 02/2018 có sai số lớn nhất, nhưng chỉ có 2.525 mẫu và không bao trùm cả tháng. Phân rã này mô tả độ ổn định theo thời gian, không thay thế đánh giá ở một giai đoạn độc lập.",small=True)

    page("6.11. Hai framework và chi phí thực nghiệm")
    rows=[]
    for name,metrics in all_metrics.items():
        for f,m in metrics.items():
            rows.append([name,f,m["completed_epochs"],m["best_epoch"],dec(m["train_seconds"],1),dec(m["train_seconds"]/m["completed_epochs"],1)])
    table(["Dataset","Framework","Epoch chạy","Epoch chọn","Tổng giây","Giây/epoch"],rows,[110,83,65,65,80,80],caption="Thời gian epoch hoàn tất gồm huấn luyện và tính validation.")
    total=sum(m["train_seconds"] for group in all_metrics.values() for m in group.values())
    p(f'Tổng thời gian được ghi trong history của bốn lượt là {dec(total/60,2)} phút. Trong lần chạy này, Keras mất khoảng {dec(retail["keras"]["train_seconds"]/retail["pytorch"]["train_seconds"],2)} lần thời gian PyTorch trên Retailrocket và {dec(stock["keras"]["train_seconds"]/stock["pytorch"]["train_seconds"],2)} lần trên S&P 500. Đây là phép đo tại máy và phiên bản cụ thể, không phải xếp hạng tốc độ phổ quát.')
    p("Runtime ghi nhận CUDA cho cả bốn lượt, PyTorch 2.5.1+cu121 và TensorFlow 2.10.1. PyTorch dùng chuỗi đóng gói theo độ dài; Keras dùng mặt nạ độ dài. Cách tổ chức phép tính khác nhau dù kiến trúc và trọng số khởi tạo tương ứng.")
    a=paired["retailrocket"];b=paired["sp500"]
    table(["Đối chiếu trên cùng ID test","Giá trị"],[["Tỷ lệ nhãn hành vi trùng nhau (%)",pct(a["prediction_agreement"])],["Sai khác xác suất trung bình",dec(a["mean_probability_difference"],8)],["Sai khác xác suất lớn nhất",dec(a["max_probability_difference"],8)],["Sai khác return trung bình",dec(b["mean_return_difference"],10)],["Sai khác return lớn nhất",dec(b["max_return_difference"],10)]],[340,143],caption="So sánh dự đoán sau huấn luyện của PyTorch và Keras.")
    p("Sai khác trung bình nhỏ không bảo đảm mọi mẫu đều giống nhau: chênh lệch xác suất lớn nhất ở Retailrocket gần 0,24. Macro-F1 hai lượt chỉ lệch khoảng 0,00032, còn RMSE cổ phiếu lệch khoảng 2,28×10<super>-8</super>. Với một seed, không có cơ sở tuyên bố ưu thế thống kê của một framework.",small=True)

    page("6.12. Kiểm chứng đầu ra và khả năng truy vết")
    verification=read("results/metric_verification.json")
    rows=[]
    for d in ["retailrocket","sp500"]:
        for f in ["pytorch","keras"]:
            r=read(f"results/checkpoint_{d}_{f}.json")
            rows.append([d,f,r["samples"],dec(r["max_abs_error"],8)])
    table(["Dataset","Framework","Mẫu tải lại","Sai khác lớn nhất"],rows,[135,105,100,143],caption="Nạp checkpoint tốt nhất trong tiến trình mới và đối chiếu dự đoán đã lưu.")
    p(f'Cả {len(verification)} lượt vượt qua kiểm tra tính lại metric từ outputs.npz và nhãn gốc. ID, thời điểm, nhãn và dự đoán trong CSV được đối chiếu với mảng dữ liệu; hash metadata được kiểm tra với cấu hình đã lưu. Epoch chọn, epoch dừng và tổng thời gian được tái dựng từ history.')
    p("Kiểm tra checkpoint dùng 256 mẫu đầu test của mỗi lượt, không khẳng định đã chạy lại suy luận toàn bộ test. Sai khác lớn nhất bằng 0 trên các batch kiểm tra trong cùng họ thiết bị. Điều này bổ sung bằng chứng về tính nhất quán của tệp đã lưu, không thay thế kiểm tra chất lượng dự báo.")
    table(["Nhóm tệp","Vai trò"],[["metrics.json; history.csv","Chỉ số test và diễn biến từng epoch"],["outputs.npz; predictions.csv","Dự đoán kèm ID, nhãn và thời điểm"],["config.json; state.json; checkpoints","Cấu hình, lựa chọn epoch và trọng số/optimizer"],["experiment_summary.csv; baselines.json","Bảng tổng hợp và các đối chứng"],["provenance_verification.json","Kiểm tra lịch sử, nguồn dữ liệu và CSV"]],[240,243],caption="Các bằng chứng đi cùng notebook và báo cáo.")
    p("Notebook lưu đầu ra thực thi, chia cell ngắn theo từng bước. README trình bày môi trường và cách tái lập, còn báo cáo tập trung vào phương pháp, số liệu, diễn giải và giới hạn. Dữ liệu gốc được dẫn bằng liên kết Kaggle và mã tải, tách khỏi repository mã nguồn.",small=True)

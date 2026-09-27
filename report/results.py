from report_engine import *
def result_pages():
    baseline=read("results/baselines.json")
    for name,label in [("retailrocket","Retailrocket"),("sp500","S&P 500")]:
        metrics={f:read(f"results/runs/{name}_{f}_seed42/metrics.json") for f in ["pytorch","keras"]}
        page("CHƯƠNG 6. KẾT QUẢ: "+label if name=="retailrocket" else "6.3. Kết quả: "+label,chapter=6 if name=="retailrocket" else None)
        keys=["accuracy","macro_f1","log_loss"] if name=="retailrocket" else ["mae","rmse","directional_accuracy"]
        rows=[[f]+[dec(m[k],6) for k in keys] for f,m in metrics.items()]
        rows += [[r["model"]]+[dec(r[k],6) for k in keys] for r in baseline if r["dataset"]==name]
        table(["Mô hình"]+keys,rows,[135,116,116,116],caption="Kết quả test của RNN và đối chứng; một seed cho mỗi framework.")
        fig(name+"_learning.png","Đường học của hai framework; loss train và metric validation có đơn vị được ghi riêng.",maxheight=210)
        target="macro_f1" if name=="retailrocket" else "rmse"
        best=(max if name=="retailrocket" else min)(metrics,key=lambda f:metrics[f][target])
        p(f'Trong hai lượt RNN, {best} có {target} tốt hơn theo số đo quan sát. Thứ hạng này thuộc một seed và một tập chia; không thể suy ra ưu thế thống kê chung của framework.')
        p("Các đối chứng chỉ dùng thống kê train. Chênh lệch được đọc trên cùng ID test. Thời gian và số epoch có thể khác vì early stopping chọn theo lịch sử validation của từng lượt.")
        page("6.2. Phân tích dự đoán hành vi" if name=="retailrocket" else "6.4. Phân tích dự báo cổ phiếu")
        if name=="retailrocket":
            for f in ["pytorch","keras"]:
                fig(name+"_"+f+"_confusion.png","Confusion matrix "+f+"; màu chuẩn hóa theo hàng, số ghi là số mẫu.",maxheight=225)
            p("Các lỗi được phân biệt theo hành vi thật để xem khả năng phát hiện cart và transaction. Accuracy tổng thể không đủ mô tả hai lớp hiếm. Ma trận này không đo tỷ lệ chuyển đổi của khách không còn sự kiện tiếp theo.",small=True)
        else:
            fig(name+"_pytorch_forecast.png","PyTorch: giá và return AAPL trên giai đoạn test; mã minh họa được chọn trước.",maxheight=260)
            fig(name+"_keras_forecast.png","Keras: cùng mã và thời gian; dự đoán một phiên quan sát tiếp theo.",maxheight=260)
            p("Mỗi dự đoán dùng giá cuối cửa sổ thật. Đây là đánh giá một bước, không phải cuộn dự đoán nhiều bước mà không cập nhật quan sát.",small=True)
    page("6.5. Chi phí, checkpoint và kiểm chứng")
    rows=[]
    for name in ["retailrocket","sp500"]:
        for f in ["pytorch","keras"]:
            m=read(f"results/runs/{name}_{f}_seed42/metrics.json")
            rows.append([name,f,m["completed_epochs"],m["best_epoch"],dec(m["train_seconds"],1)])
    table(["Dataset","Framework","Epoch chạy","Epoch chọn","Giây train"],rows,[120,95,88,88,92],caption="Chi phí quan sát trên thiết bị và số epoch được lựa chọn.")
    verification=read("results/metric_verification.json")
    p(f'{len(verification)} lượt được tính lại metric từ dự đoán và nhãn đã lưu. Việc tải lại checkpoint được kiểm tra trong tiến trình mới trên một batch theo kích thước gốc. Hồ sơ chi tiết nằm trong results.')
    p("Thời gian là tổng thời gian epoch đã hoàn tất, gồm tính validation, chưa bao gồm mọi bước tải dữ liệu và xuất báo cáo. Các ứng dụng nền có thể ảnh hưởng phép đo trên laptop. Không chuẩn hóa tốc độ theo số tham số đơn thuần.")

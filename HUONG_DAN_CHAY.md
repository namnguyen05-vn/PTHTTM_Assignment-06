# Xem kết quả và tái chạy ASG 06

**Bốn lượt huấn luyện đã hoàn tất. Không cần chạy lại để xem kết quả.**

## Mở bộ bài

1. Báo cáo cuối: report/Assignment06_NguyenNgocHoangNam_B23DCCN585.pdf.
2. Mở START_JUPYTER.bat và chọn kernel Python (Assignment 06).
3. Notebook 08 chứa bảng so sánh và biểu đồ; notebook 09 chứa hồ sơ kiểm chứng.
4. Kết quả gốc nằm trong results/runs; bảng tổng hợp ở results/experiment_summary.csv.
5. Dữ liệu được giữ ở E:\PTHTTM\ASG_06_data; mã nguồn ở E:\PTHTTM\ASG_06.

RUN_FULL=False trong notebook để đọc kết quả và chạy ví dụ nhỏ. Trọng số và kết quả cuối đã có sẵn.

## Xuất lại báo cáo

FINALIZE_REPORT.bat tính lại metric, kiểm tra checkpoint, tạo đối chứng và biểu đồ, thực thi notebook rồi xuất PDF. Công cụ không huấn luyện RNN đầy đủ. PDF sau khi thay đổi nội dung vẫn cần kiểm tra bố cục trực tiếp.

## Tiếp tục một lượt chưa hoàn tất

1. Đóng kernel và ứng dụng nặng không dùng.
2. Mở RUN_TRAINING.bat; chương trình bỏ qua những lượt đã hoàn tất.
3. Nếu cần Sleep, mở PAUSE_TRAINING.bat.
4. Đợi cửa sổ báo epoch đã lưu và quá trình đã dừng, rồi mới Sleep.
5. Khi quay lại, mở RUN_TRAINING.bat để tiếp tục.

Ctrl+C giữa epoch khiến phần epoch chưa lưu phải chạy lại. Giữ nguyên checkpoint, dữ liệu và cấu hình của lượt đang chạy.

## Thực nghiệm mới

Không xóa kết quả của bộ bài đã hoàn tất để thử cấu hình mới. Dùng thư mục kết quả riêng và đánh giá cấu hình bằng validation; không chọn tham số theo test đã xem.

README.md có hướng dẫn cài môi trường trên ổ E, tải lại dữ liệu và cấu trúc đầu ra.

# Bắt đầu huấn luyện ASG 06

Dữ liệu và môi trường đã chuẩn bị. Không cần cài lại hoặc tải lại trên máy hiện tại.

1. Vào E:\PTHTTM\ASG_06.
2. Đóng kernel notebook và ứng dụng nặng không dùng.
3. Mở **RUN_TRAINING.bat**.
4. Giữ máy thức, cắm nguồn trong khi huấn luyện.
5. Chương trình chạy 4 cấu hình, tự lưu và bỏ qua lượt đã hoàn tất.

## Khi cần Sleep

1. Mở **PAUSE_TRAINING.bat**.
2. Đợi cửa sổ huấn luyện báo epoch đã lưu và quá trình đã dừng.
3. Sau đó Sleep hoặc tắt máy.
4. Khi quay lại, mở **RUN_TRAINING.bat** để tiếp tục.

Ctrl+C giữa epoch sẽ làm phần epoch chưa lưu chạy lại; các epoch đã lưu vẫn được giữ.

## Xem bài

- Bản thảo: report/Assignment06_BanThao_TruocHuanLuyen.pdf.
- Mở notebook: **START_JUPYTER.bat**.
- Cell bật huấn luyện đầy đủ mặc định **RUN_FULL=False**.
- Chưa chạy finalize trong lúc đang huấn luyện.

## Sau khi hoàn tất

Báo lại “đã hoàn tất huấn luyện ASG 06”. Kết quả nằm sẵn trong E:\PTHTTM\ASG_06\results\runs, không cần gửi lại dataset nếu vẫn làm trên máy này.

Nếu gặp lỗi, giữ nguyên kết quả và gửi nội dung lỗi. Không xóa checkpoint hoặc chuẩn bị dữ liệu lại khi một lượt đang chạy.

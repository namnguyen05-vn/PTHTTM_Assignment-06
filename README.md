# Assignment 06 — RNN và dữ liệu chuỗi

**Nguyễn Ngọc Hoàng Nam — B23DCCN585 — D23CQCN11-B — Nhóm 05**  
Giảng viên: **Trần Đình Quế**.  
Repository: https://github.com/namnguyen05-vn/PTHTTM_Assignment-06  
Email tác giả commit: **namnguyen260805@gmail.com**.

## Trạng thái

**Bốn lượt huấn luyện đầy đủ đã hoàn tất.** Metric được tính lại từ dự đoán và nhãn test; checkpoint tốt nhất của mỗi lượt được nạp trong tiến trình mới và khớp dự đoán trên batch kiểm tra. Notebook đã thực thi lại để lưu kết quả thực nghiệm.

- [Báo cáo PDF hoàn chỉnh - 56 trang](report/Assignment06_NguyenNgocHoangNam_B23DCCN585.pdf): bìa theo ASG 04, có lý thuyết, phân tích dữ liệu, kết quả, đối chứng, biểu đồ và kết luận.
- [Mười notebook chia cell nhỏ](notebooks/).
- [Bảng kết quả tổng hợp](results/experiment_summary.csv).
- [Kết quả, lịch sử và checkpoint của bốn lượt](results/runs/).
- [Hướng dẫn xem bài và chạy lại](HUONG_DAN_CHAY.md).

Phạm vi chỉ có **Simple RNN**, không triển khai LSTM/GRU: 2 dataset × 2 framework = **4 lượt chính**, seed 42. Bản thảo trước huấn luyện còn được giữ để truy vết, không phải báo cáo nộp cuối.

| Dataset | Framework | Chỉ số test chính | Epoch chọn / đã chạy |
|---|---|---:|---:|
| Retailrocket | PyTorch | Macro-F1 0,582107 | 8 / 12 |
| Retailrocket | Keras | Macro-F1 0,581783 | 8 / 12 |
| S&P 500 | PyTorch | RMSE return 0,0155328805 | 1 / 5 |
| S&P 500 | Keras | RMSE return 0,0155328577 | 1 / 5 |

Retailrocket: RNN cải thiện macro-F1 và AP lớp hiếm, nhưng accuracy thấp hơn đối chứng luôn-view và log loss cao hơn Markov. S&P 500: RNN chưa vượt zero-return hoặc trung bình train về sai số; RMSE của trung bình train là 0,0152914985. Không diễn giải đường giá một bước bám sát thực tế như bằng chứng về lợi nhuận hoặc dự báo vượt đối chứng.

Tổng thời gian epoch ghi trong history khoảng **32,41 phút**, gồm validation. Chỉ có một seed và một tập chia; chưa có kiểm định ưu thế thống kê giữa framework.

## 1. Môi trường và tái chạy trên máy hiện tại

- Mã nguồn: E:\PTHTTM\ASG_06.
- Dữ liệu: E:\PTHTTM\ASG_06_data.
- Python đã kiểm tra: E:\PTHTTM\ASG_04\ASG_04_runtime\env\Scripts\python.exe.
- Môi trường dựa trên Anaconda đã có PyTorch 2.5.1+cu121 và TensorFlow/Keras 2.10.1/2.10.0. **Không cần cài lại trên máy hiện tại.**
- Các BAT đặt thư mục tạm, cache tải gói và dữ liệu Jupyter trên ổ E.

Các lượt chính đã hoàn tất, không cần huấn luyện lại để xem bài. Khi cần tiếp tục một lượt chưa hoàn tất, đóng kernel và ứng dụng nặng không dùng, rồi mở **RUN_TRAINING.bat**. Thứ tự: Retailrocket/PyTorch, Retailrocket/Keras, S&P 500/PyTorch, S&P 500/Keras.

Mỗi lượt dùng toàn bộ train đã chuẩn bị, tối đa 20 epoch và có early stopping. Lượt hoàn tất được bỏ qua; hàng đợi có khóa để tránh chạy trùng.

Các tệp đầu ra và PDF cuối đã nằm sẵn trên máy. RUN_TRAINING.bat bỏ qua kết quả đã hoàn tất; không xóa kết quả hoặc đổi cấu hình trong cùng thư mục để chạy một thí nghiệm mới.

## 2. Tạm dừng, Sleep và tiếp tục

1. Mở **PAUSE_TRAINING.bat** để yêu cầu dừng sau epoch hiện tại.
2. Đợi cửa sổ huấn luyện xác nhận epoch đã lưu và quá trình đã dừng, rồi mới Sleep.
3. Sau khi mở máy, chạy **RUN_TRAINING.bat**. Chương trình khôi phục trọng số, optimizer và epoch.
4. Nếu dùng Ctrl+C hoặc bị tắt giữa epoch, phần epoch chưa lưu phải chạy lại; các epoch đã lưu được giữ.
5. Không chạy hai lượt cùng thư mục kết quả trong hai cửa sổ.

Hồ sơ **results/resume_verification.json** đối chiếu hai epoch chạy liền với dừng sau epoch một và tiếp tục ở tiến trình mới. Cả hai framework khớp đầu ra trên smoke subset trong môi trường hiện tại. Không bảo đảm khớp từng bit khi đổi thiết bị hoặc phiên bản.

## 3. Notebook

Mở **START_JUPYTER.bat**, chọn kernel **Python (Assignment 06)**. Kernel được đăng ký dưới runtime/jupyter trên ổ E.

| Notebook | Nội dung |
|---|---|
| 00 | Hàm NumPy, trạng thái ẩn, BPTT và gradient |
| 01–02 | Khảo sát dữ liệu, tập chia và cửa sổ |
| 03–04 | RNN Retailrocket bằng PyTorch/Keras |
| 05–06 | RNN S&P 500 bằng PyTorch/Keras |
| 07 | Cấu hình, trạng thái và tiếp tục checkpoint |
| 08 | So sánh và visualization sau huấn luyện |
| 09 | Hồ sơ kiểm chứng |

Chạy cell từ trên xuống. Ví dụ mô hình/gradient chạy CPU trên batch nhỏ. **RUN_FULL=False** không tự huấn luyện; nếu bật, cell chạy riêng cấu hình tương ứng. File BAT thuận tiện hơn khi cần tạm dừng và tránh giữ RAM của kernel minh họa.

Mã dùng cho lượt đầy đủ nằm trong src/backends.py, src/data.py và src/training.py. Hàm trong notebook minh họa cùng kiến trúc. Khi thử thuật toán mới, sửa nguồn chung và dùng thư mục kết quả mới; sửa cell minh họa không tự đổi bộ huấn luyện nguồn.

## 4. Dataset và nhiệm vụ

### Retailrocket

Nguồn: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset  
Giấy phép nguồn: CC BY-NC-SA 4.0.

events.csv có **2.756.101 sự kiện**. Sau loại trùng và timestamp mơ hồ, còn **2.750.872 sự kiện**:

| Train | Validation | Test |
|---:|---:|---:|
| 747.842 | 122.917 | 118.192 |

Một mẫu dùng tối đa 20 hành vi đã biết để dự đoán loại hành vi kế tiếp: view, addtocart, transaction. Chỉ phiên có sự kiện kế tiếp tạo nhãn; đây không phải xác suất mua của toàn bộ khách truy cập. Phiên mới khi đổi khách hoặc ngắt quá 30 phút. Phiên cắt ranh giới tập chia bị loại; khách quay lại vẫn có thể xuất hiện ở nhiều giai đoạn.

Các dòng cùng khách và timestamp nhưng khác thông tin bị loại để tránh áp đặt thứ tự giả. Quy tắc này ảnh hưởng một phần giao dịch nhiều sản phẩm; audit báo rõ số lượng. Trường raw_visitors trong metadata hiện đếm visitor **sau làm sạch**, không phải số visitor trong raw.

### S&P 500

Nguồn: https://www.kaggle.com/datasets/camnugent/sandp500  
Giấy phép nguồn: CC0.

all_stocks_5yr.csv có **619.040 dòng**, **505 mã**. Loại 23 dòng không hợp lệ; giữ 619.017 dòng. Cửa sổ 30 phiên tạo:

| Train | Validation | Test |
|---:|---:|---:|
| 414.013 | 94.249 | 95.007 |

Đầu ra là log return phiên quan sát kế tiếp; giá dự báo suy ra từ giá cuối cửa sổ đã biết. Chỉ dùng giá/khối lượng đến phiên trước target. Chia theo ngày toàn cục, không trộn ngẫu nhiên train/test, không nối hai mã.

Thẻ nguồn không xác lập đầy đủ chính sách corporate action theo thời điểm. Thành phần chỉ số, giá lịch sử và khoảng thiếu giới hạn khả năng suy rộng. Return lớn được giữ, không loại theo chất lượng dự đoán.

### Chuẩn bị lại dữ liệu

Máy hiện tại đã chuẩn bị xong. Trên máy khác:

~~~bat
set ASG06_DATA_DIR=E:\PTHTTM\ASG_06_data
python tools/download_data.py
python tools/prepare_data.py
python tools/check_data.py
python tools/figures.py
~~~

Nếu Kaggle yêu cầu xác thực, tải riêng hai CSV. Có thể lưu nội dung CSV thuần dưới tên raw/retailrocket.download và raw/sp500.download, hoặc ZIP một tệp; trình đọc kiểm tra cấu trúc. Không đưa token Kaggle vào Git.

## 5. Cài mới Anaconda trên ổ E

Chỉ dùng khi chưa có môi trường; không cần thực hiện lại trên máy đã kiểm tra. Tạo thư mục cache/tmp trước, rồi chạy trong Anaconda Prompt:

~~~bat
set CONDA_PKGS_DIRS=E:\PTHTTM\conda_pkgs
set PIP_CACHE_DIR=E:\PTHTTM\ASG_06_data\cache\pip
set TEMP=E:\PTHTTM\ASG_06_data\cache\tmp
set TMP=%TEMP%
conda env create --prefix E:\PTHTTM\ASG_06_runtime\env -f environment.yml
conda activate E:\PTHTTM\ASG_06_runtime\env
set ASSIGNMENT06_PYTHON=E:\PTHTTM\ASG_06_runtime\env\python.exe
START_JUPYTER.bat
~~~

SETUP_ENV.bat ưu tiên ASSIGNMENT06_PYTHON; nếu chưa đặt, dùng runtime ASG 04 đã có. Chạy BAT từ cửa sổ đã đặt biến khi muốn chọn môi trường khác.

GPU Windows tương ứng:

~~~bat
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
set ASG06_CUDA_DIR=%CONDA_PREFIX%\Library\bin
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
~~~

Hai framework dùng CUDA phù hợp riêng và chạy trong tiến trình tách nhau; cần driver tương thích. Phiên bản thực tế ghi ở results/runtime_versions.json; environment.yml khóa các gói chính. Môi trường hiện tại đã chạy kiểm tra, chưa tuyên bố cài sạch thành công trên mọi máy. Tham khảo [TensorFlow](https://www.tensorflow.org/install/pip) và [PyTorch](https://pytorch.org/get-started/previous-versions/).

## 6. Lệnh chạy và kiểm chứng

~~~bat
python run_experiments.py --smoke
python run_experiments.py --resume
python run_experiments.py --dataset retailrocket --framework pytorch --resume
~~~

Smoke chỉ dùng tối đa 512 mẫu train, 128 mẫu validation, hai epoch; đầu ra riêng ở results/smoke, không đánh giá test chính thức. Dùng --cpu nếu muốn chạy CPU.

~~~bat
python tools/check_data.py
python tools/verify_backend.py pytorch
python tools/verify_backend.py keras
python tools/verify_resume.py
python tools/verify_results.py
python tools/analyze_results.py
~~~

Sau khi đủ bốn lượt chính, mở FINALIZE_REPORT.bat (tự đăng ký kernel trên ổ E), hoặc chạy trong môi trường có kernel assignment06:

~~~bat
python tools/finalize.py
~~~

Bước này tính lại metric, tải checkpoint trong tiến trình mới, tạo đối chứng và hình, thực thi notebook, xuất PDF. Nó không huấn luyện RNN. PDF vẫn cần kiểm tra bố cục trực tiếp. Giữ RUN_FULL=False trong notebook khi chỉ muốn xuất báo cáo; công cụ kiểm tra điều kiện này trước khi thực thi notebook.

## 7. Đầu ra và tái lập

Mỗi results/runs/dataset_framework_seed42 có:

- config.json: thiết lập và hash metadata dữ liệu.
- runtime.json: backend, thiết bị, tham số học.
- state.json: epoch đã lưu, checkpoint tốt nhất và early stopping.
- checkpoints/: checkpoint cuối và tốt nhất, gồm optimizer.
- history.csv: loss, validation và thời gian từng epoch.
- outputs.npz: ID, nhãn, thời gian và đầu ra mô hình.
- predictions.csv: dự đoán đọc được; chứng khoán có giá thực và giá suy ra.
- metrics.json: chỉ xuất hiện cuối cùng khi lượt hoàn tất.

Output Retailrocket là logits; output chứng khoán là return đã chia thang. Nhân target_scale trong metadata để về log return gốc; predictions.csv đã quy đổi. Xác suất hành vi lấy bằng softmax. Giá suy ra giới hạn log return ±20 để tránh overflow; số dự đoán chạm giới hạn được ghi trong metric.

Hai framework dùng cùng dữ liệu, trọng số ban đầu, loss, class weights, cửa sổ và thứ tự batch; không khẳng định mọi cập nhật số học khớp bitwise. Chỉ có một seed: không báo cáo độ lệch chuẩn nhiều lượt hoặc ưu thế thống kê của framework.

Dataset gốc, cache, token và sách giảng viên không nằm trong Git. Source báo cáo, notebook, audit và hồ sơ kiểm chứng được lưu để truy vết.

## 8. Hồ sơ sau huấn luyện

Metric được đối chiếu trên toàn bộ test; kiểm tra nạp checkpoint dùng 256 mẫu đầu mỗi lượt. CSV được so với NPZ, ID, nhãn và timestamp; lịch sử early stopping được dựng lại theo min_delta và patience.

- results/metric_verification.json: bốn lượt tính lại metric.
- results/checkpoint_*.json: nạp lại trọng số và đối chiếu dự đoán.
- results/provenance_verification.json: cấu hình, hash metadata, CSV và lựa chọn epoch.
- results/paired_framework_analysis.json: sai khác dự đoán giữa hai framework.
- results/stock_monthly_metrics.csv: sai số từng tháng, không dùng để chọn lại mô hình.
- results/final_verification.json: kiểm tra bộ bài và PDF cuối.

results/preparation_verification.json lưu trạng thái lịch sử trước huấn luyện; trạng thái cuối được ghi trong final_verification.json.

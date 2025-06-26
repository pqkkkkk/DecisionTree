# Đồ án 2 - Decision Tree - Môn Cở sở trí tuệ nhân tạo

## Mô tả đồ án

Dự án này thực hiện phân tích dữ liệu, tiền xử lý và xây dựng mô hình dự đoán trên ba bộ dữ liệu:
- **Bank Marketing** ([bank/](bank/)): Dự đoán khách hàng có đăng ký tiền gửi có kỳ hạn không.
- **Heart Disease** ([heart+disease/](heart+disease/)): Dự đoán nguy cơ mắc bệnh tim.
- **Penguins** ([penguins/](penguins/)): Phân loại loài chim cánh cụt.

Các notebook chính:
- [bank.ipynb](bank.ipynb): Xử lý và dự đoán trên bộ dữ liệu Bank.
- [heart_disease.ipynb](heart_disease.ipynb): Xử lý và dự đoán trên bộ dữ liệu Heart Disease.
- [penguins.ipynb](penguins.ipynb): Xử lý và dự đoán trên bộ dữ liệu Penguins.

Các hàm tiền xử lý dữ liệu được định nghĩa trong [process.py](process.py).

## Yêu cầu

- Python 3.8+
- Các thư viện trong [requirements.txt](requirements.txt)

## Cài đặt

1. **Clone repository** (nếu cần):

    ```sh
    git clone <https://github.com/pqkkkkk/DecisionTree.git>
    cd <DecisionTree>
    ```
2. **Cài đặt Graphviz về máy tính của bạn**
- Bạn có thể tải xuống từ [trang chủ Graphviz](https://graphviz.org/download/).
- Thêm đường dẫn đến thư mục chứa file dot.exe vào biến môi trường PATH của hệ điều hành.
    Ví dụ trên Windows:
    - Tải xuống và cài đặt Graphviz.
    - Thêm đường dẫn đến thư mục `bin` của Graphviz (ví dụ: `C:\Program Files\Graphviz\bin`) vào biến môi trường PATH.
3. **Cài đặt các thư viện cần thiết:**

    ```sh
    pip install -r requirements.txt
    ```

## Hướng dẫn chạy

1. **Chạy các notebook:**

    - Mở từng file notebook (`bank.ipynb`, `heart_disease.ipynb`, `penguins.ipynb`) bằng Jupyter Notebook hoặc Visual Studio Code.
    - Chạy tuần tự từng cell để thực hiện tiền xử lý, huấn luyện mô hình và đánh giá kết quả.

2. **Tiền xử lý dữ liệu:**
    - Các hàm tiền xử lý đã được định nghĩa trong [`penguins_data_preprocessing`](process.py#L30) và [`bank_data_preprocessing`](process.py#L53) ([process.py](process.py)).
    - Dữ liệu sẽ được tự động xử lý khi chạy các notebook.

3. **Kết quả:**
    - Kết quả mô hình, biểu đồ độ chính xác, ma trận nhầm lẫn sẽ được hiển thị trực tiếp trong notebook.

## Thư mục dữ liệu

- **bank/**: Chứa dữ liệu ngân hàng.
- **heart+disease/**: Chứa dữ liệu bệnh tim.
- **penguins/**: Chứa dữ liệu chim cánh cụt.


> Vui lòng đảm bảo các file dữ liệu gốc nằm đúng vị trí như cấu trúc thư mục để notebook hoạt động chính xác.
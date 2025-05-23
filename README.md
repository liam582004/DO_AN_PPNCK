# 📘 Đồ án môn học: Phương pháp nghiên cứu khoa học (PPNCKH)

## 🌟 Giới thiệu dự án

Đây là repo chứa toàn bộ nội dung đồ án môn học **Phương pháp nghiên cứu khoa học**.  
Repo được tổ chức thành các thư mục tương ứng với các bài tập hàng tuần do giảng viên giao, các tài liệu phục vụ nghiên cứu, cũng như file ghi lại quá trình làm việc của từng thành viên trong nhóm.

---

## 🎯 Giới thiệu dự án

### 🧠 Phân Loại Cảm Xúc Văn Bản Sử Dụng LSTM và Tokenization

Dự án này tập trung vào việc phân loại cảm xúc trong các câu tiếng Anh bằng cách sử dụng mô hình mạng nơ-ron hồi tiếp LSTM kết hợp với kỹ thuật tokenization. Việc nhận diện cảm xúc là một tác vụ quan trọng trong Xử lý ngôn ngữ tự nhiên (NLP), đặc biệt hữu ích trong các ứng dụng như phân tích phản hồi khách hàng, chatbot, và theo dõi cảm xúc người dùng.

### 🎯 Mục tiêu

- Xây dựng mô hình học sâu để phân loại cảm xúc từ văn bản.
- Ứng dụng LSTM để khai thác thông tin tuần tự trong dữ liệu ngôn ngữ.
- Sử dụng tokenization và embedding để chuyển văn bản thành dạng số phục vụ huấn luyện.

### 🛠️ Công nghệ sử dụng

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn (phân tích & trực quan hóa dữ liệu)
- NLTK / Keras Tokenizer (tiền xử lý ngôn ngữ)

### 📊 Dữ liệu

Bộ dữ liệu bao gồm các câu tiếng Anh được gán nhãn với 6 loại cảm xúc:

- Vui 😄
- Buồn 😢
- Giận dữ 😠
- Ngạc nhiên 😲
- Yêu thương 😄

### 🧪 Quy trình thực hiện

1. **Tiền xử lý văn bản**: Làm sạch, token hóa, padding chuỗi.
2. **Mã hóa nhãn**: Chuyển các cảm xúc thành dạng one-hot encoding.
3. **Xây dựng mô hình**: Thiết kế mô hình LSTM có lớp embedding.
4. **Huấn luyện**: Đào tạo mô hình với dữ liệu đã xử lý.
5. **Đánh giá & dự đoán**: Kiểm tra độ chính xác và dự đoán cảm xúc từ câu mới.

### 🚀 Kết quả đạt được

# Phân Loại Cảm Xúc Đánh Giá Sản Phẩm Sử Dụng Mô Hình LSTM

Dự án này triển khai một mô hình học sâu LSTM để phân loại cảm xúc trong các đánh giá sản phẩm thành 5 loại cảm xúc chính: **Giận dữ, Vui vẻ, Yêu thương, Buồn bã, Ngạc nhiên**.

## 📁 Dữ liệu
 Amazon Product Reviews 2023 (truy cập tại https://amazon-reviews-2023.github.io), 
- **Tập huấn luyện**: 561136 mẫu, chiếm ~80% tổng dữ liệu] 
- **Tập kiểm tra**: 70142 mẫu, chiếm ~10% tổng dữ liệu
- **Tập validation**: 70143 mẫu, chiếm ~10% tổng dữ liệu 

### 📌 Cách gán nhãn:
Dựa vào số sao người dùng đánh giá sản phẩm:
- 1 sao → Giận dữ (Anger)
- 2 sao → Buồn bã (Sadness)
- 3 sao → Ngạc nhiên (Surprise)
- 4 sao → Vui vẻ (Joy)
- 5 sao → Yêu thương (Love)

> ⚠️ **Lưu ý**: Cách gán nhãn theo số sao chỉ mang tính chất ước lượng và không hoàn toàn chính xác với cảm xúc thực sự trong nội dung đánh giá.

## 🧠 Kiến trúc mô hình

- **Embedding layer**: sử dụng từ điển GloVe tiền huấn luyện
- **LSTM layer**: học biểu diễn tuần tự của văn bản
- **Dense layer**: đầu ra với hàm kích hoạt Softmax
- **Loss function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Batch size**: 32
- **Epochs**: 8

## 📊 Kết quả huấn luyện

![Đồ thị độ chính xác và hàm mất mát](images/c06c6534-1410-490c-ad24-207c9c109584.png)


- **Độ chính xác trên tập huấn luyện**: > 0.95
- **Độ chính xác trên tập validation**: ~0.90 - 0.92
- **Loss**: giảm đều trong quá trình huấn luyện

### 🧪 Đánh giá trên tập kiểm tra

- **Số mẫu kiểm tra**: 70,143
- **Micro avg Accuracy**: 0.15
- **Weighted avg F1-score**: 0.06

### 📋 Báo cáo phân loại:

| Cảm xúc    | Precision | Recall | F1-score | Số mẫu |
|------------|-----------|--------|----------|--------|
| Giận dữ    | 0.15      | 0.94   | 0.26     | 10,133 |
| Vui vẻ     | 0.18      | 0.07   | 0.10     | 7,994  |
| Yêu thương | 0.58      | 0.01   | 0.02     | 42,004 |
| Buồn bã    | 0.09      | 0.01   | 0.03     | 4,301  |
| Ngạc nhiên | 0.06      | 0.00   | 0.01     | 5,711  |

- Mô hình có xu hướng **dự đoán quá mức nhãn "Giận dữ"**, dẫn đến **Recall cao nhưng Precision thấp**
- Hiệu suất thấp ở các cảm xúc như **Yêu thương**, **Buồn bã**, và **Ngạc nhiên** do:
  - **Mất cân bằng dữ liệu**
  - **Gán nhãn cảm xúc không chính xác từ số sao**
  - **Văn bản đánh giá có thể mơ hồ, nhiều tầng ý nghĩa**

### 📉 Ma Trận Nhầm Lẫn

Dưới đây là ma trận nhầm lẫn trên tập kiểm tra cho 5 lớp cảm xúc đã chọn:

![Ma Trận Nhầm Lẫn](./b00ec8d8-6f70-45a5-a38d-2c14ff476be9.png)

### So sách với các nghiên cứu trước
- Mô hình đạt weighted F1-score ~0.06, thấp hơn nhiều so với các nghiên cứu sử dụng bộ dữ liệu chuẩn và nhãn thủ công.
- Nguyên nhân do nhãn suy luận chưa chuẩn, dữ liệu mất cân bằng và mô hình đơn giản.
- Mục tiêu không phải là cạnh tranh độ chính xác, mà là khảo nghiệm quy trình và khai thác dữ liệu thực tế.

### Ý nghĩa thực tiễn
- Cung cấp khung tham khảo cho các dự án xử lý dữ liệu review sản phẩm không có nhãn cảm xúc.
- Làm rõ thách thức và giới hạn của việc tạo nhãn tự động.
- Gợi ý hướng phân tích xu hướng cảm xúc trên tập dữ liệu lớn.
- Nền tảng để phát triển và cải tiến mô hình trong các nghiên cứu tiếp theo.


## 👥 Thành viên nhóm 9

| Họ và tên         | Email                                           | GitHub                                         | Website cá nhân                                  |
|-------------------|-------------------------------------------------|------------------------------------------------|-------------------------------------------------|
| Hồ Hưng Lộc       | [hohungloc58@gmail.com](mailto:hohungloc58@gmail.com) | [github.com/liam582004](https://github.com/liam582004) | [liam582004.github.io/portfolio-cv](https://liam582004.github.io/portfolio-cv/) |
| Nguyễn Trường Sinh | [emailcuasinh@gmail.com](mailto:emailcuasinh@gmail.com) | [github.com/SN1PE7](https://github.com/SN1PE7) | [sn1pe7.github.io](https://sn1pe7.github.io/) |
| Hoàng Sỹ Khiêm    | [sgu3121410263@gmail.com](sgu3121410263@gmail.com) | [github.com/khiemHoang141](https://github.com/khiemHoang1410) | [khiemhoang1410.github.io](https://khiemhoang1410.github.io) |




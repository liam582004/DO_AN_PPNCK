# Phân Loại Cảm Xúc Văn Bản Sử Dụng LSTM (Text Emotion Detection using LSTM)

Nhóm 9
Hoàng Sỹ Khiêm	    - 3121410263
Hồ Hưng Lộc	    - 3122410219 
Nguyễn Trường Sinh  - 3122410358 

Dự án này tập trung vào việc xây dựng và đánh giá một mô hình học sâu dựa trên kiến trúc Long Short-Term Memory (LSTM) để tự động phân loại cảm xúc từ nội dung văn bản tiếng Anh. Dự án bao gồm quy trình hoàn chỉnh từ xử lý dữ liệu thô (đánh giá sản phẩm), tiền xử lý văn bản, xây dựng mô hình, huấn luyện, đánh giá, và cung cấp một script để thực hiện dự đoán trên dữ liệu mới.

---

##  Mục Lục
- [Tổng Quan](#tổng-quan)
- [Nguồn Dữ Liệu](#nguồn-dữ-liệu)
- [Quy Trình Thực Hiện](#quy-trình-thực-hiện)
  - [Chuẩn Bị và Tiền Xử Lý Dữ Liệu](#chuẩn-bị-và-tiền-xử-lý-dữ-liệu)
  - [Xây Dựng và Huấn Luyện Mô Hình](#xây-dựng-và-huấn-luyện-mô-hình)
  - [Đánh Giá Mô Hình](#đánh-giá-mô-hình)
- [Cấu Trúc Thư Mục Dự Án](#cấu-trúc-thư-mục-dự-án)
- [Yêu Cầu Hệ Thống và Thư Viện](#yêu-cầu-hệ-thống-và-thư-viện)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
  - [Huấn Luyện Lại Mô Hình](#huấn-luyện-lại-mô-hình)
  - [Sử Dụng Model Đã Huấn Luyện Để Dự Đoán](#sử-dụng-model-đã-huấn-luyện-để-dự-đoán)
- [Kết Quả Dự Kiến](#kết-quả-dự-kiến)
- [Hạn Chế và Hướng Phát Triển](#hạn-chế-và-hướng-phát-triển)
- [Đóng Góp](#đóng-góp)
- [Giấy Phép](#giấy-phép)
- [Lời Cảm Ơn](#lời-cảm-ơn)

---

## Tổng Quan (Overview)

Mục tiêu chính của dự án là phát triển một hệ thống có khả năng nhận diện và phân loại các trạng thái cảm xúc phổ biến (ví dụ: `love`, `joy`, `surprise`, `sadness`, `anger`) từ các đoạn văn bản đánh giá sản phẩm bằng tiếng Anh. Phương pháp tiếp cận chính là sử dụng mạng LSTM, một kiến trúc học sâu hiệu quả cho việc xử lý dữ liệu tuần tự như ngôn ngữ tự nhiên.

Dự án cung cấp:
-   Mã nguồn Python cho việc xử lý dữ liệu, xây dựng, huấn luyện và đánh giá mô hình trong Jupyter Notebook (`Text Emotion Classifier.ipynb`).
-   Các script Python hỗ trợ (`process_raw_reviews.py`, `split_processed_data.py`, các script tạo biểu đồ và đánh giá).
-   Một script Python (`text_emo_detection.py`) để tải mô hình đã huấn luyện và thực hiện dự đoán cảm xúc trên văn bản đầu vào mới.
-   Các tài nguyên đã được huấn luyện (ví dụ: `emotion_recognizer.h5` cho model và `tokenizer.pkl` cho bộ từ vựng) cho phép người dùng sử dụng ngay mà không cần huấn luyện lại từ đầu (nếu có sẵn và được cung cấp).

---

## Nguồn Dữ Liệu (Dataset)

Dự án này sử dụng dữ liệu đánh giá sản phẩm tiếng Anh từ bộ dữ liệu công khai **Amazon Product Reviews 2023** (truy cập tại: [https://amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io)).

-   **Dữ liệu thô:** Được cung cấp dưới dạng file JSON Lines, chứa các thông tin như điểm xếp hạng (`rating`), tiêu đề (`title`), và nội dung đánh giá (`text`).
-   **Xử lý và tạo nhãn:**
    -   Tiêu đề và nội dung đánh giá được kết hợp thành một trường văn bản duy nhất.
    -   Điểm xếp hạng (1-5 sao) được sử dụng để suy luận và gán nhãn cho 5 lớp cảm xúc:
        -   `rating > 4.5`: `love`
        -   `rating > 3.5`: `joy`
        -   `rating > 2.5`: `surprise`
        -   `rating > 1.5`: `sadness`
        -   `rating <= 1.5`: `anger`
    -   Dữ liệu sau đó được làm sạch (loại bỏ mã `[[VIDEOID...]]`, tag HTML, ký tự đặc biệt) và lưu dưới định dạng `văn_bản;nhãn_cảm_xúc`.
-   **Phân chia dữ liệu:** Bộ dữ liệu đã xử lý được chia thành tập huấn luyện (training), tập kiểm định (validation), và tập kiểm thử (testing) với tỷ lệ ví dụ: 80%-10%-10%. Các file dữ liệu này (ví dụ: `reviews_train.txt`, `reviews_val.txt`, `reviews_test.txt`) được lưu trong thư mục `dataset/`.

---

## Quy Trình Thực Hiện (Workflow)

Quy trình tổng thể của dự án bao gồm các bước chính sau:

### 1. Chuẩn Bị và Tiền Xử Lý Dữ Liệu
   -   **Xử lý dữ liệu JSON Lines thô:** Trích xuất các trường cần thiết (`rating`, `title`, `text`), kết hợp `title` và `text`.
   -   **Làm sạch văn bản thô:** Loại bỏ các yếu tố gây nhiễu như mã ID video, thẻ HTML, ký tự xuống dòng, dấu chấm phẩy trong nội dung.
   -   **Ánh xạ Rating sang Nhãn Cảm xúc:** Sử dụng logic đã định nghĩa để gán nhãn cảm xúc (`love`, `joy`, `surprise`, `sadness`, `anger`) cho từng đánh giá.
   -   **Tiền xử lý văn bản chuyên sâu (NLP):** (Thực hiện trong `Text Emotion Classifier.ipynb`)
        -   Chuyển văn bản thành chữ thường.
        -   Loại bỏ dấu câu (`string.punctuation`).
        -   Tách từ (Tokenization) bằng `nltk.tokenize.word_tokenize`.
        -   Loại bỏ từ dừng (Stopwords) tiếng Anh bằng `nltk.corpus.stopwords`.
        -   Đưa từ về dạng gốc (Lemmatization) bằng `nltk.stem.WordNetLemmatizer`.
   -   **Mã hóa Nhãn:** Sử dụng `sklearn.preprocessing.LabelEncoder` để chuyển nhãn cảm xúc dạng text sang dạng số.

### 2. Xây Dựng và Huấn Luyện Mô Hình (Trong `Text Emotion Classifier.ipynb`)
   -   **Tokenization và Vector hóa:**
        -   Sử dụng `Tokenizer` từ `tensorflow.keras.preprocessing.text` để xây dựng từ điển từ vựng và chuyển đổi văn bản thành chuỗi các chỉ số số nguyên.
        -   Đệm các chuỗi (Padding Sequences) bằng `pad_sequences` để đảm bảo tất cả các chuỗi đầu vào có cùng độ dài (`maxlen`).
   -   **Kiến trúc Mô hình LSTM:**
        -   Xây dựng một mô hình tuần tự (`Sequential`) bằng Keras API.
        -   Lớp `Embedding`: Để học các vector biểu diễn từ.
        -   Lớp `LSTM`: Để nắm bắt các phụ thuộc tuần tự và ngữ cảnh trong văn bản.
        -   Lớp `Dense` (Output): Với hàm kích hoạt `softmax` để phân loại đa lớp cảm xúc.
   -   **Biên dịch và Huấn luyện Mô hình:**
        -   Optimizer: Adam.
        -   Loss Function: `sparse_categorical_crossentropy`.
        -   Metrics: `accuracy`.
        -   Huấn luyện mô hình (`model.fit()`) trên tập huấn luyện, sử dụng tập kiểm định để theo dõi và tránh overfitting.
   -   **Lưu trữ Model và Tokenizer:** Lưu mô hình đã huấn luyện (`.h5`) và đối tượng Tokenizer (`.pkl`).

### 3. Đánh Giá Mô Hình
   -   Đánh giá hiệu năng của mô hình trên tập kiểm thử bằng các chỉ số: Test Loss, Test Accuracy.
   -   Sử dụng `classification_report` từ `sklearn.metrics` để xem Precision, Recall, F1-score cho từng lớp cảm xúc.
   -   Sử dụng `confusion_matrix` từ `sklearn.metrics` và trực quan hóa bằng `matplotlib`, `seaborn` để phân tích sự nhầm lẫn giữa các lớp.

---

## Cấu Trúc Thư Mục Dự Án (Project Directory Structure)
text-emotion-classification-using-lstm-and-tokenization/
│
├── dataset/                             # Chứa các bộ dữ liệu
│   ├── reviews_train.txt                # Dữ liệu huấn luyện (ví dụ)
│   ├── reviews_val.txt                  # Dữ liệu kiểm định (ví dụ)
│   ├── reviews_test.txt                 # Dữ liệu kiểm thử (ví dụ)
│   ├── All_Beauty.jsonl          		  # Dữ liệu thô
│   ├── split_processed_data.py          # Chia train/val/test
│   └── process_raw_reviews.py           # Xử lý JSON Lines
│
├── saved_models/                        # (Đề xuất) Lưu model và tokenizer
│   ├── emotion_recognizer.h5            # File model
│   └── tokenizer.pkl                    # File tokenizer
│
├── Text Emotion Classifier.ipynb        # Notebook chính
├── text_emo_detection.py                # Script dự đoán cảm xúc
├── README.md                            # Chính là file này
└── requirements.txt                     # Các thư viện cần thiết

## Yêu Cầu Hệ Thống và Thư Viện (System Requirements and Libraries)

-   Python 3.x
-   Các thư viện Python chính:
    -   `numpy`
    -   `nltk` (với các gói `punkt`, `stopwords`, `wordnet` cần được tải về)
    -   `scikit-learn`
    -   `tensorflow` (phiên bản 2.x)
    -   `matplotlib`
    -   `seaborn`
    -   `pandas` (sử dụng trong notebook)
    -   `json`, `re`, `pickle` (thư viện chuẩn của Python)

Để cài đặt các thư viện cần thiết (ngoại trừ các thư viện chuẩn), bạn có thể tạo một file `requirements.txt` với nội dung ví dụ:
numpy
nltk
scikit-learn
tensorflow
matplotlib
seaborn
pandas

Và chạy lệnh:
```bash
pip install -r requirements.txt

pip install numpy nltk scikit-learn tensorflow matplotlib seaborn pandas

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
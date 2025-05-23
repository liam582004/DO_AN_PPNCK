import random

# --- PHẦN CẤU HÌNH ---
input_processed_file = 'data_review.txt' 
train_file_out = 'reviews_train.txt'
val_file_out = 'reviews_val.txt'
test_file_out = 'reviews_test.txt'

train_ratio = 0.8 # 80% cho training
val_ratio = 0.1   # 10% cho validation (phần còn lại 10% sẽ cho test)
# test_ratio sẽ là 1.0 - train_ratio - val_ratio

# --- HÀM CHÍNH ---
def split_data():
    print(f"Đang đọc file: {input_processed_file}")
    try:
        with open(input_processed_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file input '{input_processed_file}'.")
        return

    if not lines:
        print(f"LỖI: File input '{input_processed_file}' không có dữ liệu.")
        return
        
    random.shuffle(lines) # Xáo trộn các dòng
    print(f"Đã xáo trộn {len(lines)} dòng dữ liệu.")

    num_total_lines = len(lines)
    train_end_idx = int(num_total_lines * train_ratio)
    val_end_idx = train_end_idx + int(num_total_lines * val_ratio)

    train_lines = lines[:train_end_idx]
    val_lines = lines[train_end_idx:val_end_idx]
    test_lines = lines[val_end_idx:]

    try:
        with open(train_file_out, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        print(f"Đã ghi {len(train_lines)} dòng vào file training: {train_file_out}")

        with open(val_file_out, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        print(f"Đã ghi {len(val_lines)} dòng vào file validation: {val_file_out}")

        with open(test_file_out, 'w', encoding='utf-8') as f:
            f.writelines(test_lines)
        print(f"Đã ghi {len(test_lines)} dòng vào file test: {test_file_out}")
        
        print("\n--- HOÀN TẤT PHÂN CHIA DỮ LIỆU ---")
    except Exception as e:
        print(f"LỖI khi ghi file output: {e}")

if __name__ == '__main__':
    split_data()
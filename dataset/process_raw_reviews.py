import json
import re # Thư viện cho regular expressions (xử lý text nâng cao)

input_jsonl_file = 'All_Beauty.jsonl'  
output_txt_file = 'data_review.txt'    

# Định nghĩa cách map từ "rating" (1-5 sao) sang "nhãn cảm xúc"
def map_rating_to_emotion(rating_float):
    rating = float(rating_float) # Đảm bảo rating là số thực

    if rating > 4.5:      # Thường là 5 sao (ví dụ: 4.6 - 5.0)
        return 'love'     # "Hơn cả joy" -> yêu thích, rất hài lòng
    elif rating > 3.5:    # Thường là 4 sao (ví dụ: 3.6 - 4.5)
        return 'joy'
    elif rating > 2.5:    # Thường là 3 sao (ví dụ: 2.6 - 3.5)
        return 'surprise' 
    elif rating > 1.5:    # Thường là 2 sao (ví dụ: 1.6 - 2.5)
        return 'sadness'
    elif rating >= 0:     # Dưới 1.5 (Thường là 1 sao, hoặc 0 nếu có)
        return 'anger'
    return None

# --- KẾT THÚC PHẦN CẤU HÌNH ---

def process_raw_review_data():
    try:
        with open(input_jsonl_file, 'r', encoding='utf-8') as infile, \
             open(output_txt_file, 'w', encoding='utf-8') as outfile:

            print(f"🚀 Bắt đầu xử lý file: {input_jsonl_file}")
            line_count = 0
            processed_count = 0
            skipped_no_label_count = 0
            skipped_missing_data_count = 0
            skipped_empty_text_count = 0
            json_error_count = 0

            for line in infile:
                line_count += 1
                if line_count % 10000 == 0: 
                    print(f"Đã đọc {line_count} dòng...")
                try:
                    data_item = json.loads(line.strip())

                    rating_str = data_item.get('rating') 
                    title = data_item.get('title', '') 
                    text_content = data_item.get('text', '')

                    if rating_str is None or not text_content:
                        skipped_missing_data_count +=1
                        continue
                    
                    try:
                        rating = float(rating_str)
                    except ValueError:
                        skipped_missing_data_count += 1
                        continue

                    emotion_label = map_rating_to_emotion(rating)

                    if emotion_label:
                        combined_text = title
                        if title and text_content:
                            combined_text += ". " + text_content
                        elif text_content:
                            combined_text = text_content
                        
                        cleaned_text = combined_text.replace('\n', ' ').replace('\r', ' ')
                        cleaned_text = re.sub(r'\[\[VIDEOID:[a-zA-Z0-9_.-]+\]\]', '', cleaned_text)
                        cleaned_text = re.sub(r'<br\s*/?>', ' ', cleaned_text)
                        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
                        cleaned_text = cleaned_text.replace(';', ',')
                        cleaned_text = cleaned_text.strip()
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

                        if cleaned_text:
                            outfile.write(f"{cleaned_text};{emotion_label}\n")
                            processed_count += 1
                        else:
                            skipped_empty_text_count +=1
                    else:
                        skipped_no_label_count +=1

                except json.JSONDecodeError:
                    json_error_count +=1
                except Exception:
                    json_error_count +=1
            
            print(f"\n--- HOÀN TẤT XỬ LÝ ---")
            print(f"Tổng số dòng đã đọc từ file input: {line_count}")
            print(f"Số dòng đã xử lý thành công và ghi ra file output: {processed_count}")
            print(f"Số dòng bị bỏ qua do thiếu rating/text hoặc rating không hợp lệ: {skipped_missing_data_count}")
            print(f"Số dòng bị bỏ qua do rating không map ra nhãn: {skipped_no_label_count}")
            print(f"Số dòng bị bỏ qua do text rỗng sau khi làm sạch: {skipped_empty_text_count}")
            print(f"Số dòng bị lỗi JSON hoặc lỗi khác: {json_error_count}")
            print(f"🎉 File output đã được lưu tại: {output_txt_file}")

    except FileNotFoundError:
        print(f" LỖI: Không tìm thấy file input '{input_jsonl_file}'.")
    except Exception as e:
        print(f" LỖI TỔNG QUÁT KHÔNG XÁC ĐỊNH: {e}")

if __name__ == '__main__':
    process_raw_review_data()
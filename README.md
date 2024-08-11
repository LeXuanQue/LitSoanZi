import cv2
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import queue
import threading

# Đường dẫn đến video trên máy tính
video_path = r'C:\Users\Admin\Videos\dainam\5302702977662.mp4'

# Tải mô hình và bộ xử lý hình ảnh từ transformers
try:
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

class_names = ["class_0", "class_1", "class_2", "class_3", "class_4",
                "class_5", "class_6", "class_7", "class_8", "class_9"]

frame_count = 0
batch_frames = []
batch_size = 8
frame_queue = queue.Queue()
frame_lock = threading.Lock()  # Để đồng bộ hóa frame_count
search_frame = None
search_frame_loaded = False
search_frame_number = None  # Số frame cần tìm

def process_batch(batch_frames):
    inputs = processor(images=batch_frames, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    return logits

def read_video():
    global frame_count, search_frame, search_frame_loaded, search_frame_number
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: Could not open video.')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))  # Giảm độ phân giải
        frame_queue.put(frame)

        # Kiểm tra xem đây có phải là frame mà người dùng muốn tìm không
        if frame_count == search_frame_number:
            search_frame = frame.copy()
            search_frame_loaded = True
            print(f"Loaded frame {frame_count} for searching.")

        frame_count += 1  # Cập nhật số frame

    cap.release()
    frame_queue.put(None)  # Đánh dấu kết thúc

def process_frames():
    global frame_count, search_frame_loaded, search_frame_number
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Chỉ xử lý các frame sau khi frame tìm kiếm đã được tải
        if search_frame_loaded:
            if frame_queue.qsize() > 0:
                frame_idx = frame_queue.qsize()

            # So sánh frame hiện tại với frame tìm kiếm
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            search_image = cv2.cvtColor(search_frame, cv2.COLOR_BGR2RGB)
            search_pil_image = Image.fromarray(search_image)

            if np.array_equal(np.array(pil_image), np.array(search_pil_image)):
                print(f"Found the frame at position: {frame_count - frame_idx}")
                # Hiển thị thông tin
                display_text = f"Frame: {frame_count - frame_idx} - FOUND!"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Video Frame', frame)
                cv2.waitKey(0)
                break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        batch_frames.append(pil_image)

        if len(batch_frames) >= batch_size:
            logits = process_batch(batch_frames)
            batch_frames.clear()

            for i, logit in enumerate(logits):
                predicted_class_idx = torch.argmax(logit, dim=-1).item()
                predicted_label = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else "Unknown"

                # Thêm số thứ tự của frame và dự đoán vào frame
                display_text = f"Frame: {frame_count}, Prediction: {predicted_label}"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Hiển thị frame với OpenCV
                cv2.imshow('Video Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

def load_search_frame():
    global search_frame_number
    search_frame_number = int(input("Enter the frame number to search for: "))

# Khởi động các luồng
load_search_frame_thread = threading.Thread(target=load_search_frame)
read_thread = threading.Thread(target=read_video)
process_thread = threading.Thread(target=process_frames)

load_search_frame_thread.start()
load_search_frame_thread.join()  # Chờ cho đến khi frame tìm kiếm được tải xong

read_thread.start()
process_thread.start()

read_thread.join()
process_thread.join()

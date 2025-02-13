import cv2
import numpy as np
import random
import time
import pygame

# Các đường dẫn file (đảm bảo chúng đúng trên thiết bị hoặc được đóng gói cùng app)
drum_roll_path = r"C:\Users\Admin\Downloads\drum-roll-sound-effect-278576.mp3"
success_fanfare_path = r"C:\Users\Admin\Downloads\471815325_28282533481391020_150980357218729850_n.mp3"
prototxt_path = r"F:\tann\deploy.prototxt"
model_path    = r"F:\tann\res10_300x300_ssd_iter_140000.caffemodel"

# Khởi tạo pygame mixer để phát âm thanh
pygame.mixer.init()

# Tải model phát hiện khuôn mặt của OpenCV
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Hàm phát hiện khuôn mặt từ frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    (h, w) = processed.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(processed, (400, 400)),
                                 1.0, (400, 400),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    threshold = 0.3
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (sx, sy, ex, ey) = box.astype("int")
            sx, sy = max(0, sx), max(0, sy)
            ex, ey = min(w, ex), min(h, ey)
            faces.append((sx, sy, ex, ey))
    return faces


# ------------------------------
# Phần giao diện bằng Kivy
# ------------------------------
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button

class FaceDetectionApp(App):
    def build(self):
        # Layout chính theo chiều dọc
        root = BoxLayout(orientation='vertical')

        # Widget hiển thị camera feed (sử dụng Kivy Image)
        self.image = Image()
        root.add_widget(self.image)

        # Layout chứa 3 nút ở dưới cùng
        button_layout = BoxLayout(size_hint_y=0.2)
        start_btn = Button(text="Start Drumroll")
        reset_btn = Button(text="Reset")
        quit_btn  = Button(text="Quit")

        start_btn.bind(on_release=self.start_drumroll)
        reset_btn.bind(on_release=self.reset_state)
        quit_btn.bind(on_release=self.stop_app)

        button_layout.add_widget(start_btn)
        button_layout.add_widget(reset_btn)
        button_layout.add_widget(quit_btn)

        root.add_widget(button_layout)

        # Khởi tạo camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không mở được camera!")
            return root

        # Các biến trạng thái
        self.phase = 'idle'  # trạng thái: idle, drumroll, zoom
        self.drumroll_duration = 5.0  # thời gian drumroll (giây)
        self.drumroll_start_time = None

        self.faces_list = []     # danh sách khuôn mặt
        self.last_face_box = None  # khuôn mặt được chọn ngẫu nhiên cuối cùng

        # Hiệu ứng zoom
        self.zoom_scales = np.linspace(1.0, 2.5, 40)  # 40 bước zoom
        self.zoom_index = 0

        # Người thắng: lưu khuôn mặt và frame ban đầu
        self.winner_box = None
        self.winner_frame = None

        # Lịch cập nhật frame (30 FPS)
        Clock.schedule_interval(self.update, 1.0/30.0)

        return root

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Lưu frame gốc để sử dụng cho hiệu ứng zoom
        original_frame = frame.copy()

        # Phát hiện khuôn mặt
        faces = detect_faces(frame)
        if faces:
            self.faces_list = faces  # cập nhật danh sách

        # Nếu đang ở trạng thái idle hoặc drumroll, vẽ khung xanh quanh các khuôn mặt
        if self.phase in ['idle', 'drumroll']:
            for (sx, sy, ex, ey) in faces:
                cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)

        # Xử lý các trạng thái:
        if self.phase == 'drumroll':
            # Nếu có khuôn mặt, chọn ngẫu nhiên một khuôn mặt và lưu lại
            if self.faces_list:
                face_idx = random.randint(0, len(self.faces_list) - 1)
                (sx, sy, ex, ey) = self.faces_list[face_idx]
                self.last_face_box = (sx, sy, ex, ey)
                # Hiệu ứng nhấp nháy: vẽ vòng tròn đỏ quanh khuôn mặt được chọn
                if int(time.time() * 5) % 2 == 0:
                    cx, cy = (sx + ex) // 2, (sy + ey) // 2
                    radius = int(0.6 * max(ex - sx, ey - sy) // 2)
                    cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 3)

            # Kiểm tra thời gian drumroll
            if self.drumroll_start_time and (time.time() - self.drumroll_start_time) >= self.drumroll_duration:
                # Kết thúc drumroll
                pygame.mixer.music.stop()
                if self.last_face_box is not None:
                    self.winner_box = self.last_face_box
                    self.winner_frame = original_frame.copy()
                    self.phase = 'zoom'
                    self.zoom_index = 0
                    # Phát fanfare 1 lần
                    pygame.mixer.music.load(success_fanfare_path)
                    pygame.mixer.music.play(0)
                    print("Người thắng đã được chọn. Hiệu ứng zoom bắt đầu!")
                else:
                    print("Không có khuôn mặt nào được chọn!")
                    self.phase = 'idle'

        elif self.phase == 'zoom':
            # Hiệu ứng zoom: phóng to khuôn mặt được chọn
            if self.winner_box is not None and self.winner_frame is not None:
                (sx, sy, ex, ey) = self.winner_box
                face_crop = self.winner_frame[sy:ey, sx:ex]
                if face_crop.size != 0:
                    # Tăng dần scale
                    if self.zoom_index < len(self.zoom_scales):
                        scale = self.zoom_scales[self.zoom_index]
                        self.zoom_index += 1
                    else:
                        scale = self.zoom_scales[-1]
                    effect_frame = frame.copy()
                    resized = cv2.resize(face_crop, None, fx=scale, fy=scale)
                    rH, rW = resized.shape[:2]
                    cY, cX = effect_frame.shape[0] // 2, effect_frame.shape[1] // 2
                    startY = max(0, cY - rH // 2)
                    startX = max(0, cX - rW // 2)
                    endY = min(startY + rH, effect_frame.shape[0])
                    endX = min(startX + rW, effect_frame.shape[1])
                    effect_frame[startY:endY, startX:endX] = resized[0:(endY - startY), 0:(endX - startX)]
                    frame = effect_frame

        # Cập nhật frame lên widget Image của Kivy
        # (lưu ý: đảo ngược trục dọc để hiển thị đúng)
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    # Hàm gọi khi nhấn nút "Start Drumroll"
    def start_drumroll(self, instance):
        if self.phase == 'idle' and self.faces_list:
            self.phase = 'drumroll'
            self.drumroll_start_time = time.time()
            self.faces_list = []  # reset danh sách khuôn mặt
            self.last_face_box = None
            pygame.mixer.music.load(drum_roll_path)
            pygame.mixer.music.play(-1)
            print("Drumroll bắt đầu...")

        elif self.phase == 'idle' and not self.faces_list:
            print("Không phát hiện khuôn mặt. Hãy đảm bảo có người trong khung hình.")

    # Hàm gọi khi nhấn nút "Reset"
    def reset_state(self, instance):
        if self.phase == 'zoom':
            self.phase = 'idle'
            self.winner_box = None
            self.winner_frame = None
            self.faces_list = []
            self.zoom_index = 0
            print("Reset về trạng thái idle.")

    # Hàm gọi khi nhấn nút "Quit"
    def stop_app(self, instance):
        self.cap.release()
        self.stop()

    # Hủy tài nguyên khi app dừng lại
    def on_stop(self):
        if self.cap:
            self.cap.release()

if __name__ == '__main__':
    FaceDetectionApp().run()

import cv2
from ultralytics import YOLO
import math

# --- CẤU HÌNH ---
# Đường dẫn đến file model bạn vừa tải về
model_path = 'best.pt' 

# Ngưỡng tin cậy (Confidence Threshold): Chỉ hiện khi AI chắc chắn trên bao nhiêu %
# 0.5 nghĩa là chắc chắn 50% trở lên mới hiện
conf_threshold = 0.5 

# --- KHỞI TẠO ---
# Tải model
model = YOLO(model_path)

# Mở webcam (số 0 thường là webcam mặc định của laptop)
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Chiều rộng khung hình
cap.set(4, 720)  # Chiều cao khung hình

print("Đang khởi động Camera... Nhấn 'q' để thoát.")

while True:
    success, img = cap.read()
    if not success:
        print("Không thể đọc từ Camera.")
        break

    # --- NHẬN DIỆN (INFERENCE) ---
    # Chạy model YOLO trên khung hình hiện tại
    # stream=True giúp chạy mượt hơn với video/webcam
    results = model(img, stream=True, conf=conf_threshold)

    # --- VẼ KẾT QUẢ LÊN MÀN HÌNH ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Lấy tọa độ khung chữ nhật (Bounding Box)pip install opencv-python ultralytics.
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 2. Vẽ khung chữ nhật lên ảnh
            # Màu (255, 0, 255) là màu tím, độ dày nét là 3
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # 3. Lấy tên class (nhãn) và độ tin cậy
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = model.names[cls] # Ví dụ: 'Hello', 'A', 'B'

            # 4. Viết chữ lên ảnh
            text = f'{class_name} {confidence}'
            
            # Tính toán vị trí để chữ không bị che hoặc tràn ra ngoài
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0) # Màu xanh dương
            thickness = 2

            cv2.putText(img, text, org, font, fontScale, color, thickness)

    # Hiển thị cửa sổ webcam
    cv2.imshow('Nhan Dien Ky Hieu Tay - YOLOv8', img)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
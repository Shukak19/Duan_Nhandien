import cv2
from ultralytics import YOLO
import math
import time # Import thêm thư viện thời gian

# --- CẤU HÌNH ---
model_path = 'best.pt' 

# Ngưỡng tin cậy (Nên để cao một chút khoảng 0.6 - 0.7 để tránh ghi nhận sai lúc đang chuyển tay)
conf_threshold = 0.6 

# --- KHỞI TẠO ---
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720)  

# --- BIẾN ĐỂ GHÉP CHỮ ---
sentence = ""             # Biến lưu trữ câu hoàn chỉnh
last_time = time.time()   # Mốc thời gian lần cuối ghi nhận chữ
cooldown = 2.0            # Thời gian chờ (giây) giữa 2 lần ghi nhận. Bạn có thể chỉnh nhỏ lại (ví dụ 1.5) nếu múa tay nhanh.

print("Đang khởi động Camera...")
print("- Nhấn 'q' để thoát.")
print("- Nhấn 'c' (Clear) để xóa dòng chữ đã ghép.")

while True:
    success, img = cap.read()
    if not success:
        print("Không thể đọc từ Camera.")
        break

    # --- NHẬN DIỆN ---
    results = model(img, stream=True, conf=conf_threshold)

    # --- XỬ LÝ VÀ VẼ KẾT QUẢ ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Lấy tọa độ và vẽ khung chữ nhật
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # 2. Lấy thông tin nhãn
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = model.names[cls] 

            # 3. Viết nhãn nhỏ dính trên khung chữ nhật
            text = f'{class_name} {confidence}'
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 4. LOGIC GHÉP CHỮ (Ghi nhận vào câu)
            current_time = time.time()
            # Nếu thời gian hiện tại trừ đi lần cuối ghi nhận lớn hơn khoảng chờ (2 giây)
            if (current_time - last_time) > cooldown:
                sentence += class_name       # Cộng thêm chữ cái mới vào câu
                last_time = current_time     # Cập nhật lại mốc thời gian

    # --- IN CÂU HOÀN CHỈNH LÊN MÀN HÌNH ---
    # Vẽ một dải băng màu đen ở dưới cùng màn hình để làm nền cho chữ dễ đọc
    cv2.rectangle(img, (0, 650), (1280, 720), (0, 0, 0), -1)
    
    # In biến "sentence" lên dải băng đen đó (Chữ màu xanh lá cây, to rõ ràng)
    cv2.putText(img, f"Van ban: {sentence}", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Hiển thị
    cv2.imshow('Nhan Dien Ngon Ngu Ky Hieu - YOLO', img)

    # --- BẮT SỰ KIỆN BÀN PHÍM ---
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):  # Nếu bấm phím 'c'
        sentence = ""      # Xóa trắng câu để viết lại từ đầu

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
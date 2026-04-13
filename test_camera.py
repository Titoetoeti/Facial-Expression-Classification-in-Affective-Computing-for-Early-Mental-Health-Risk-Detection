import cv2
import sys

print("=== KIỂM TRA CAMERA ===")
print(f"OpenCV version: {cv2.__version__}")

# Thử mở camera
print("Đang mở camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở camera index 0")
    print("Thử camera index 1...")
    cap = cv2.VideoCapture(1)
    
if not cap.isOpened():
    print("❌ Không tìm thấy camera!")
    sys.exit(1)

print("✅ Camera đã sẵn sàng!")
print("Nhấn 'q' để thoát")

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Không đọc được frame!")
        break
    
    frame_count += 1
    
    # Vẽ text
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị
    cv2.imshow('Test Camera', frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Đã đóng camera!")
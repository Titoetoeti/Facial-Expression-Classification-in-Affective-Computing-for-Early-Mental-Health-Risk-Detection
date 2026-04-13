import cv2
import numpy as np
from collections import deque, Counter
import time
from datetime import datetime
import json
import os
# import urllib.request # Không dùng đến
import pandas as pd

# --- 1. CẤU HÌNH GIAO DIỆN (UI CONFIG) ---
# Màu sắc hệ thống (BGR - OpenCV dùng ngược với RGB)
UI_COLORS = {
    'bg_glass': (30, 30, 30),      # Màu nền kính
    'text_main': (255, 255, 255),  # Chữ trắng
    'text_dim': (170, 170, 170),   # Chữ xám
    'bar_bg': (60, 60, 60),        # Nền thanh bar
    'safe': (50, 255, 50),         # Xanh lá neon (An toàn)
    'warning': (0, 165, 255),      # Cam neon (Cảnh báo nhẹ)
    'danger': (50, 50, 255)        # Đỏ neon (Nguy hiểm)
}

# --- MỚI: BẢNG MÀU RIÊNG CHO TỪNG CẢM XÚC (THEO YÊU CẦU) ---
# Định dạng BGR: (Xanh dương, Xanh lá, Đỏ)
EMOTION_COLORS_MT = {
    'angry': (0, 0, 255),      # GIẬN DỮ -> ĐỎ
    'sad': (255, 0, 0),        # BUỒN -> XANH DƯƠNG
    'happy': (0, 255, 0),      # VUI -> XANH LÁ
    'fear': (128, 0, 128),     # Sợ hãi -> Tím
    'surprise': (0, 255, 255), # Ngạc nhiên -> Vàng
    'disgust': (0, 140, 255),  # Ghê tởm -> Cam
    'neutral': (200, 200, 200) # Bình thường -> Xám trắng
}

# Mapping tên tiếng Việt cho đẹp
VIETNAMESE_MAPPING = {
    'angry': 'GIAN DU', 'disgust': 'GHE TOM', 'fear': 'SO HAI',
    'happy': 'VUI VE', 'sad': 'BUON', 'surprise': 'NGAC NHIEN', 'neutral': 'BINH THUONG'
}

# --- 2. CÁC HÀM VẼ GIAO DIỆN ---
def draw_glass_panel(img, x, y, w, h, alpha=0.7):
    """Vẽ tấm kính bán trong suốt bên trái"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), UI_COLORS['bg_glass'], -1)
    cv2.line(overlay, (x+w, y), (x+w, y+h), (100, 100, 100), 1) # Viền nhẹ
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_tech_box(img, x, y, w, h, color, label):
    """Vẽ khung mặt kiểu Tech với màu sắc động"""
    l = int(w / 4) # Độ dài góc
    t = 2 # Độ dày
    # Vẽ 4 góc
    cv2.line(img, (x, y), (x + l, y), color, t)
    cv2.line(img, (x, y), (x, y + l), color, t)
    cv2.line(img, (x+w, y), (x+w-l, y), color, t)
    cv2.line(img, (x+w, y), (x+w, y+l), color, t)
    cv2.line(img, (x, y+h), (x+l, y+h), color, t)
    cv2.line(img, (x, y+h), (x, y+h-l), color, t)
    cv2.line(img, (x+w, y+h), (x+w-l, y+h), color, t)
    cv2.line(img, (x+w, y+h), (x+w, y+h-l), color, t)
    
    # Tag tên (Nền màu theo cảm xúc, chữ đen/trắng tùy độ sáng)
    cv2.rectangle(img, (x, y-25), (x+w, y), color, -1)
    # Chọn màu chữ tương phản (đơn giản hóa: dùng màu đen cho nổi)
    text_color = (0,0,0) 
    cv2.putText(img, label, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

def draw_bar(img, x, y, w, val, color, label):
    """Vẽ thanh năng lượng"""
    # Label
    cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_COLORS['text_dim'], 1)
    # Background Bar
    cv2.rectangle(img, (x, y), (x+w, y+8), UI_COLORS['bar_bg'], -1)
    # Value Bar
    fill = int(w * val)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+8), color, -1)
    # Text %
    cv2.putText(img, f"{int(val*100)}%", (x+w+5, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_COLORS['text_main'], 1)

# --- 3. LOGIC CODE GỐC (GIỮ NGUYÊN) ---
try:
    from fer import FER
    print("[INFO] Đang dùng FER library (pre-trained)")
except ImportError:
    print("[ERROR] Chưa cài fer. Đang cài đặt...")
    os.system('pip install fer tensorflow pandas opencv-python')
    from fer import FER

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
THRESHOLDS = {
    'depression': {'sad_ratio': 0.60, 'negative_ratio': 0.70, 'happy_ratio': 0.10},
    'anxiety': {'fear_ratio': 0.50, 'negative_ratio': 0.60},
    'stress': {'angry_ratio': 0.50, 'negative_ratio': 0.50},
    'apathy': {'neutral_ratio': 0.90}
}

class EmotionBuffer:
    def __init__(self, window_seconds=30, fps=5):
        self.window_size = window_seconds * fps
        self.buffer = deque(maxlen=self.window_size)
        self.all_records = []
    
    def add(self, emotion, confidence):
        timestamp = time.time()
        record = {
            'emotion': emotion, 'confidence': confidence,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        self.buffer.append(record)
        self.all_records.append(record)
    
    def get_statistics(self):
        if not self.buffer: return None
        emotions = [item['emotion'] for item in self.buffer]
        counts = Counter(emotions)
        total = len(emotions)
        stats = {}
        for e in EMOTIONS:
            stats[f'{e}_ratio'] = counts.get(e, 0) / total
        
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        positive_emotions = ['happy', 'surprise']
        stats['negative_ratio'] = sum(counts.get(e, 0) for e in negative_emotions) / total
        stats['positive_ratio'] = sum(counts.get(e, 0) for e in positive_emotions) / total
        stats['neutral_ratio'] = counts.get('neutral', 0) / total
        stats['dominant_emotion'] = counts.most_common(1)[0][0]
        return stats
    
    def export_to_csv(self, filename='emotion_history.csv'):
        if not self.all_records: return
        df = pd.DataFrame(self.all_records)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Đã export {len(self.all_records)} records → {filename}")

class MentalHealthAnalyzer:
    def __init__(self):
        self.history = []
        self.alerts = []
    
    def analyze(self, stats):
        if not stats: return None
        risks, risk_score = [], 0
        
        if (stats['sad_ratio'] > THRESHOLDS['depression']['sad_ratio'] and
            stats['negative_ratio'] > THRESHOLDS['depression']['negative_ratio']):
            risks.append('Depression')
            risk_score += 40
        if (stats['fear_ratio'] > THRESHOLDS['anxiety']['fear_ratio']):
            risks.append('Anxiety')
            risk_score += 30
        if stats['angry_ratio'] > THRESHOLDS['stress']['angry_ratio']:
            risks.append('Stress')
            risk_score += 20
        if stats['neutral_ratio'] > THRESHOLDS['apathy']['neutral_ratio']:
            risks.append('Apathy')
            risk_score += 15
        
        if risk_score >= 60: risk_level = 'HIGH'
        elif risk_score >= 30: risk_level = 'MEDIUM'
        elif risk_score > 0: risk_level = 'LOW'
        else: risk_level = 'NORMAL'
        
        result = {
            'timestamp': datetime.now(), 'risk_level': risk_level,
            'risk_score': risk_score, 'potential_issues': risks, 'statistics': stats
        }
        self.history.append(result)
        if risk_level in ['HIGH', 'MEDIUM']: self.alerts.append(result)
        return result
    
    def get_mental_health_score(self, stats):
        if not stats: return 50
        score = 100
        score -= stats['sad_ratio'] * 40
        score -= stats['fear_ratio'] * 30
        score -= stats['angry_ratio'] * 20
        score += stats['happy_ratio'] * 20
        return max(0, min(100, score))

# --- 4. HÀM MAIN ---
def main():
    print("="*60)
    print("MENTAL HEALTH SYSTEM - COLOR UPDATE")
    print("="*60)
    
    # LƯU Ý: Nếu máy lag, đổi mtcnn=True thành mtcnn=False
    emotion_detector = FER(mtcnn=False) 
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    
    cap.set(3, 1280)
    cap.set(4, 720)
    
    buffer = EmotionBuffer(window_seconds=30, fps=5)
    analyzer = MentalHealthAnalyzer()
    
    frame_count = 0
    current_box = None
    current_label = ""
    current_color = EMOTION_COLORS_MT['neutral'] # Màu mặc định
    last_stats = None
    last_result = None
    last_score = 50
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        frame_count += 1
        
        # --- A. PHẦN XỬ LÝ ---
        if frame_count % 3 == 0:
            try:
                result = emotion_detector.detect_emotions(frame)
                if result:
                    face = max(result, key=lambda x: x['box'][2] * x['box'][3])
                    current_box = face['box']
                    
                    emos = face['emotions']
                    dom_emo = max(emos, key=emos.get)
                    conf = emos[dom_emo]
                    
                    buffer.add(dom_emo, conf)
                    current_label = VIETNAMESE_MAPPING.get(dom_emo, dom_emo)
                    
                    # --- MỚI: Lấy màu dựa trên cảm xúc hiện tại ---
                    current_color = EMOTION_COLORS_MT.get(dom_emo, EMOTION_COLORS_MT['neutral'])

                    last_stats = buffer.get_statistics()
                    if last_stats:
                        last_result = analyzer.analyze(last_stats)
                        last_score = analyzer.get_mental_health_score(last_stats)
                else:
                    if frame_count % 15 == 0: current_box = None
            except: pass

        # --- B. PHẦN VẼ UI ---
        frame = draw_glass_panel(frame, 0, 0, 360, h_frame)
        
        cv2.putText(frame, "DANH GIA CAM XUC ", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        cv2.putText(frame, "MADE BY QUOCCONG (AND GEMINI)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_COLORS['text_main'], 1)
        
        if last_stats:
            # Điểm số
            cv2.putText(frame, "HEALTH SCORE", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_COLORS['text_dim'], 1)
            s_color = UI_COLORS['safe']
            if last_score < 70: s_color = UI_COLORS['warning']
            if last_score < 40: s_color = UI_COLORS['danger']
            cv2.putText(frame, f"{last_score:.1f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, s_color, 3)
            
            # --- CÁC THANH ĐO (ĐÃ ĐỔI MÀU THEO YÊU CẦU) ---
            y_start = 240
            # Buồn -> Xanh dương
            draw_bar(frame, 20, y_start, 200, last_stats['sad_ratio'], EMOTION_COLORS_MT['sad'], "Ty le Buon (Depression)")
            # Sợ hãi -> Tím (Giữ nguyên cho khác biệt)
            draw_bar(frame, 20, y_start+40, 200, last_stats['fear_ratio'], EMOTION_COLORS_MT['fear'], "Ty le Lo au (Anxiety)")
            # Giận dữ -> Đỏ
            draw_bar(frame, 20, y_start+80, 200, last_stats['angry_ratio'], EMOTION_COLORS_MT['angry'], "Ty le Stress")
            # Vui vẻ -> Xanh lá
            draw_bar(frame, 20, y_start+120, 200, last_stats['happy_ratio'], EMOTION_COLORS_MT['happy'], "Ty le Vui ve")
            
            # Hộp cảnh báo
            status_text = "TRANG THAI: ON DINH"
            box_color = UI_COLORS['safe']
            
            if last_result and last_result['potential_issues']:
                status_text = f"CANH BAO: {last_result['potential_issues'][0].upper()}"
                box_color = UI_COLORS['danger']
                if (frame_count // 10) % 2 == 0:
                     cv2.rectangle(frame, (10, h_frame-100), (350, h_frame-20), box_color, 2)
                     cv2.putText(frame, "WARNING!", (30, h_frame-65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            else:
                cv2.rectangle(frame, (10, h_frame-100), (350, h_frame-20), UI_COLORS['safe'], 1)
            
            cv2.putText(frame, status_text.replace("CANH BAO: ", ""), (30, h_frame-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLORS['text_main'], 2)
            
        else:
            cv2.putText(frame, "Dang thu thap du lieu...", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLORS['text_dim'], 1)

        # 5. Vẽ Khung mặt Tech (Dùng màu động current_color)
        if current_box is not None:
            bx, by, bw, bh = current_box
            draw_tech_box(frame, bx, by, bw, bh, current_color, current_label)

        cv2.imshow('Mental Health AI System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
             filename = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
             cv2.imwrite(filename, frame)
             print(f"Saved: {filename}")
        elif key == ord('e'):
             buffer.export_to_csv()

    buffer.export_to_csv()
    
    if analyzer.history:
        report = {
            'summary': {'total_analyses': len(analyzer.history)},
            'analyses': [{'timestamp': str(i['timestamp']), 'risk': i['risk_level'], 'issues': i['potential_issues']} for i in analyzer.history]
        }
        with open('mental_health_report.json', 'w') as f: json.dump(report, f, indent=2)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
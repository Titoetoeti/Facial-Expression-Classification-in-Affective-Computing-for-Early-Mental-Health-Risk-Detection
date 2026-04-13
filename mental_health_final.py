import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from collections import deque, Counter
import time
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
import urllib.request

# Cấu hình
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NEGATIVE_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Sad']
POSITIVE_EMOTIONS = ['Happy', 'Surprise']

EMOTION_COLORS = {
    'Angry': (0, 0, 255), 'Disgust': (0, 128, 128), 'Fear': (128, 0, 128),
    'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Surprise': (0, 255, 255),
    'Neutral': (128, 128, 128)
}

THRESHOLDS = {
    'depression': {'sad_ratio': 0.60, 'negative_ratio': 0.70, 'happy_ratio': 0.10},
    'anxiety': {'fear_ratio': 0.50, 'negative_ratio': 0.60},
    'stress': {'angry_ratio': 0.50, 'negative_ratio': 0.50},
    'apathy': {'neutral_ratio': 0.90}
}

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128*6*6, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class EmotionBuffer:
    def __init__(self, window_seconds=30, fps=10):
        self.window_size = window_seconds * fps
        self.buffer = deque(maxlen=self.window_size)
    def add(self, emotion, confidence):
        self.buffer.append({'emotion': emotion, 'confidence': confidence, 'timestamp': time.time()})
    def get_statistics(self):
        if not self.buffer: return None
        emotions = [item['emotion'] for item in self.buffer]
        confidences = [item['confidence'] for item in self.buffer]
        counts = Counter(emotions)
        total = len(emotions)
        stats = {f'{e}_ratio': counts.get(e, 0)/total for e in EMOTIONS}
        stats['negative_ratio'] = sum(counts.get(e, 0) for e in NEGATIVE_EMOTIONS)/total
        stats['positive_ratio'] = sum(counts.get(e, 0) for e in POSITIVE_EMOTIONS)/total
        stats['neutral_ratio'] = counts.get('Neutral', 0)/total
        stats['avg_confidence'] = np.mean(confidences)
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        stats['emotion_variance'] = changes/total if total > 1 else 0
        stats['dominant_emotion'] = counts.most_common(1)[0][0]
        return stats
    def get_features_vector(self):
        stats = self.get_statistics()
        return [stats['Sad_ratio'], stats['Fear_ratio'], stats['Angry_ratio'], 
                stats['Happy_ratio'], stats['negative_ratio'], stats['positive_ratio'],
                stats['emotion_variance'], stats['avg_confidence']] if stats else None

class MentalHealthAnalyzer:
    def __init__(self):
        self.history, self.alerts = [], []
    def analyze(self, stats):
        if not stats: return None
        risks, risk_score = [], 0
        if (stats['Sad_ratio'] > 0.6 and stats['negative_ratio'] > 0.7 and stats['Happy_ratio'] < 0.1):
            risks.append('Depression'); risk_score += 40
        if stats['Fear_ratio'] > 0.5 or (stats['Fear_ratio'] + stats['Sad_ratio']) > 0.6:
            risks.append('Anxiety'); risk_score += 30
        if stats['Angry_ratio'] > 0.5:
            risks.append('Stress'); risk_score += 20
        if stats['neutral_ratio'] > 0.9:
            risks.append('Apathy'); risk_score += 15
        risk_level = 'HIGH' if risk_score >= 60 else 'MEDIUM' if risk_score >= 30 else 'LOW' if risk_score > 0 else 'NORMAL'
        result = {'timestamp': datetime.now(), 'risk_level': risk_level, 
                 'risk_score': risk_score, 'potential_issues': risks, 'statistics': stats}
        self.history.append(result)
        if risk_level in ['HIGH', 'MEDIUM']: self.alerts.append(result)
        return result
    def get_mental_health_score(self, stats):
        if not stats: return 50
        score = 100 - stats['Sad_ratio']*40 - stats['Fear_ratio']*30 - stats['Angry_ratio']*20 - stats['Disgust_ratio']*15
        if stats['neutral_ratio'] > 0.8: score -= 20
        score += stats['Happy_ratio']*20
        return max(0, min(100, score))

class EmotionClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters, self.kmeans, self.scaler, self.data, self.labels = n_clusters, None, StandardScaler(), [], []
    def add_sample(self, features):
        if features: self.data.append(features)
    def fit(self):
        if len(self.data) < self.n_clusters: return None
        X_scaled = self.scaler.fit_transform(np.array(self.data))
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = self.kmeans.fit_predict(X_scaled)
        return self.labels
    def visualize(self, save_path='clustering_result.png'):
        if len(self.data) < self.n_clusters: return
        X = np.array(self.data)
        plt.figure(figsize=(10, 6))
        colors = ['green', 'yellow', 'red']
        for i in range(self.n_clusters):
            cluster_data = X[self.labels == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 4], c=colors[i], label=f'Cluster {i}', alpha=0.6)
        plt.xlabel('Sad Ratio'); plt.ylabel('Negative Ratio')
        plt.title('Emotion Clustering - Mental Health Detection')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu: {save_path}")

def main():
    print("="*70 + "\nTEMPORAL EMOTION PROFILING FOR MENTAL HEALTH DETECTION")
    print("Phát hiện dấu hiệu Trầm cảm & Lo âu qua Hồ sơ Cảm xúc Thời gian\n" + "="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(7).to(device); model.eval()
    print(f"\n[INFO] Device: {device}\n[INFO] Model: SimpleCNN\n[WARNING] Model chưa train\n")
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)), transforms.Grayscale(1),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    
    # Download Haar Cascade nếu chưa có
    cascade_file = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_file):
        print("[INFO] Đang tải Haar Cascade...")
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_file)
        print(f"[INFO] Đã tải: {cascade_file}")
    
    face_cascade = cv2.CascadeClassifier(cascade_file)
    if face_cascade.empty():
        print("[ERROR] Không load được Haar Cascade!"); return
    print("[INFO] Đã load Haar Cascade!")
    
    emotion_buffer = EmotionBuffer(30, 10)
    analyzer = MentalHealthAnalyzer()
    clusterer = EmotionClusterer(3)
    
    print("[INFO] Đang mở camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Không mở được camera!"); return
    
    cap.set(3, 800); cap.set(4, 600)
    print("[INFO] Camera sẵn sàng!\n\nPhím: q=Thoát | s=Screenshot | r=Reset | a=Analyze | v=Visualize\n")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_tensor = transform(Image.fromarray(face_roi)).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(face_tensor)
                probs = torch.nn.functional.softmax(outputs, 1)
                confidence, predicted = probs.max(1)
                emotion = EMOTIONS[predicted.item()]
                conf_score = confidence.item()
            emotion_buffer.add(emotion, conf_score)
            color = EMOTION_COLORS[emotion]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"{emotion}: {conf_score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if frame_count % 100 == 0:
            stats = emotion_buffer.get_statistics()
            if stats:
                result = analyzer.analyze(stats)
                score = analyzer.get_mental_health_score(stats)
                clusterer.add_sample(emotion_buffer.get_features_vector())
                print(f"\n[ANALYSIS #{len(analyzer.history)}] Risk: {result['risk_level']} | Score: {score:.1f}/100")
                if result['potential_issues']: print(f"Issues: {', '.join(result['potential_issues'])}")
        
        stats = emotion_buffer.get_statistics()
        if stats:
            score = analyzer.get_mental_health_score(stats)
            color = (0,255,0) if score>=70 else (0,255,255) if score>=50 else (0,0,255)
            cv2.putText(frame, f"Score: {score:.1f}/100", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Neg: {stats['negative_ratio']*100:.1f}%", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(frame, f"Pos: {stats['positive_ratio']*100:.1f}%", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Buffer: {len(emotion_buffer.buffer)}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        cv2.imshow('Mental Health Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            fn = f"ss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, frame); print(f"✅ Saved: {fn}")
        elif key == ord('r'):
            emotion_buffer = EmotionBuffer(30, 10); print("✅ Reset")
        elif key == ord('a'):
            if len(clusterer.data)>=3:
                clusterer.fit(); print(f"\n✅ Clustered {len(clusterer.data)} samples")
            else: print("\n⚠️ Need more data")
        elif key == ord('v'): clusterer.visualize()
    
    cap.release(); cv2.destroyAllWindows()
    print(f"\n{'='*70}\nSUMMARY: {frame_count} frames | {len(analyzer.history)} analyses | {len(analyzer.alerts)} alerts")
    
    if analyzer.history:
        report = {'summary': {'analyses': len(analyzer.history), 'alerts': len(analyzer.alerts)},
                 'data': [{'time': str(i['timestamp']), 'risk': i['risk_level'], 'score': i['risk_score'], 
                          'issues': i['potential_issues']} for i in analyzer.history]}
        with open('report.json', 'w') as f: json.dump(report, f, indent=2)
        print("✅ Saved: report.json\n")

if __name__ == "__main__": main()
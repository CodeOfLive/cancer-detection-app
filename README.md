# 🔬 Histopatolojik Hücre Analizi (Kanser Risk Tahmini)

Derin öğrenme tabanlı bu prototip, histopatolojik doku görüntülerini analiz ederek tahmini kanser risk seviyesi üretir. MobileNetV2 transfer learning mimarisi ile eğitilen model, görüntüyü patch'lere bölerek bölgesel inceleme yapar ve asenkron işleme altyapısı ile kullanıcı deneyimini optimize eder.

🔗 **Canlı Demo:** [https://cancer-detection-app-otpt.onrender.com](https://cancer-detection-app-otpt.onrender.com)  
 **GitHub Repo:** [https://github.com/CodeOfLive/cancer-detection-app](https://github.com/CodeOfLive/cancer-detection-app)

## ✨ Özellikler
- 🧠 **Patch Bazlı Analiz:** Görüntüyü 224x224 parçalara ayırarak bölgesel risk tespiti
- ⚡ **Asenkron İşleme:** Uzun süren ML işlemlerini arka planda yöneten Job Queue sistemi
- 🔄 **Otomatik Polling:** Frontend polling ile kullanıcıyı bilgilendirme ve bağlantı kopmalarını yönetme
- 📱 **Responsive UI:** Modern glassmorphism tasarımı, mobil uyumlu arayüz
- 🛡️ **Güvenlik & Optimizasyon:** Dosya tipi kontrolü, boyut limiti, otomatik görsel küçültme

## 🛠️ Tech Stack
| Katman | Teknoloji |
|--------|-----------|
| Backend | Python 3.10, Flask, Gunicorn |
| ML/AI | TensorFlow 2.15, Keras, MobileNetV2, OpenCV, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript (Fetch API, Async/Await) |
| Deployment | Render (Free Tier), GitHub (Auto-Deploy) |
| Veri | LC25000 Histopathology Dataset (Colon ACA / Colon Normal) |

## 📦 Kurulum (Local Geliştirme)
```bash
# 1. Repoyu klonla
git clone https://github.com/CodeOfLive/cancer-detection-app.git
cd cancer-detection-app

# 2. Sanal ortam oluştur ve aktif et
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Uygulamayı başlat
python app.py
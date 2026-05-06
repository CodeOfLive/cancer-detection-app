from flask import Flask, request, jsonify, render_template
import os
import uuid
from pathlib import Path
from predict import predict_image_patches

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/upload", methods=["POST"])
def upload_image():
    import time
    start_time = time.time()
    
    if "image" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400
    
    ext = Path(file.filename).suffix.lower()
    # .webp desteği eklendi
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return jsonify({"error": "Sadece JPG/JPEG/PNG/WebP formatları desteklenir"}), 400
    
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)
    
    try:
        print(f"🔄 Analiz başlıyor: {filename}")
        result = predict_image_patches(save_path)
        result["image_path"] = f"/static/uploads/{filename}"
        print(f"✅ Analiz tamamlandı: {time.time() - start_time:.2f} sn")
        return jsonify(result)
    except Exception as e:
        print(f"❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analiz hatası: {str(e)}"}), 500
@app.route("/disclaimer")
def disclaimer():
    return """
    <div style="padding:20px; font-family:sans-serif;">
        <h2>⚠️ Yasal Uyarı</h2>
        <p><strong>Bu sistem tıbbi teşhis koymaz, sadece tahmini risk analizi yapar.</strong></p>
        <p>Gerçek klinik kullanımda uzman onayı ve yasal regülasyon gereklidir.</p>
        <p>Bu proje tamamen akademik/öğrenme amaçlı geliştirilmektedir.</p>
        <a href="/">← Ana Sayfaya Dön</a>
    </div>
    """

if __name__ == "__main__":
    # Render PORT ortam değişkenini kullan, yoksa 5000'e düş
    port = int(os.environ.get("PORT", 5000))
    # Production'da debug=False olmalı ama local test için True bırakabilirsin
    app.run(debug=False, host="0.0.0.0", port=port)
import os, uuid, traceback
from flask import Flask, request, jsonify, render_template
from pathlib import Path

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
    if "image" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi"}), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return jsonify({"error": "Sadece JPG/JPEG/PNG/WebP desteklenir"}), 400
    
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)
    
    try:
        from predict import predict_image_patches  # Import burada, lazy load
        result = predict_image_patches(save_path)
        result["image_path"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        traceback.print_exc()  # Hatayı loglara yazdır
        return jsonify({"error": f"Analiz hatası: {str(e)}"}), 500

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Sunucu meşgul. 10 sn sonra tekrar deneyin."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
import os, uuid, traceback, threading, time
from flask import Flask, request, jsonify, render_template
from pathlib import Path
from datetime import datetime, timedelta

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Job Queue (İş kuyruğu)
jobs = {}

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
    
    # Job oluştur
    job_id = str(uuid.uuid4())
    filename = f"{job_id}{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)
    
    jobs[job_id] = {
        "status": "processing",
        "created_at": datetime.now(),
        "result": None,
        "error": None
    }
    
    # Arka planda işle (Thread)
    def process_in_background():
        try:
            from predict import predict_image_patches
            result = predict_image_patches(save_path)
            result["image_path"] = f"/static/uploads/{filename}"
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "completed"
            print(f"✅ Job {job_id} tamamlandı!")
        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "failed"
            print(f"❌ Job {job_id} hatası: {e}")
            traceback.print_exc()
    
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    # Hemen job_id döndür
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job bulunamadı"}), 404
    
    job = jobs[job_id]
    
    # 15 dakikadan eski job'ları temizle
    if datetime.now() - job["created_at"] > timedelta(minutes=15):
        del jobs[job_id]
        return jsonify({"error": "Job süresi doldu"}), 410
    
    return jsonify({
        "status": job["status"],
        "result": job["result"],
        "error": job["error"]
    })

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Sunucu hatası"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
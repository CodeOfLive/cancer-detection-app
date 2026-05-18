import os, uuid, traceback, threading, time, json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_talisman import Talisman
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
import cv2  # ✅ OpenCV ile daha robust dosya okuma
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32).hex())

# 🔹 PostgreSQL (Neon) Optimizasyonu
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "connect_args": {"sslmode": "require", "connect_timeout": 10}
}

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.static_folder = 'static'
app.static_url_path = '/static'

# 🔒 Talisman CSP
csp = {
    'default-src': "'self'",
    'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    'script-src': ["'self'", "'unsafe-inline'"],
    'img-src': ["'self'", "data:", "https://image.qwenlm.ai"],
    'font-src': ["'self'", "https://fonts.googleapis.com", "https://fonts.gstatic.com"],
    'connect-src': "'self'"
}
Talisman(app, force_https=False, content_security_policy=csp)

# 🛡️ CSRF Koruması
csrf = CSRFProtect(app)
db = SQLAlchemy(app)

# ️ MODELLER
class AdminUser(db.Model):
    __tablename__ = "admins"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ImageUpload(db.Model):
    __tablename__ = "uploads"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255))
    file_size_kb = db.Column(db.Integer)
    mime_type = db.Column(db.String(50))
    ip_address = db.Column(db.String(45))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    risk_level = db.Column(db.String(20))
    cancer_ratio = db.Column(db.Float)
    total_patches = db.Column(db.Integer)
    cancer_patches = db.Column(db.Integer)
    processing_time_ms = db.Column(db.Integer)
    status = db.Column(db.String(20), default="completed")

class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 🔐 ADMIN PANEL
class SecureAdminIndexView(AdminIndexView):
    @expose("/")
    def index(self):
        if not session.get("admin_logged_in"):
            return redirect(url_for(".login"))
        return super().index()
    @expose("/login", methods=["GET", "POST"])
    def login(self):
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            user = AdminUser.query.filter_by(username=username).first()
            if user and user.check_password(password) and user.is_active:
                session["admin_logged_in"] = True
                session["admin_user"] = user.username
                try:
                    log = AuditLog(action="ADMIN_LOGIN", ip_address=request.remote_addr)
                    db.session.add(log)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"⚠️ Audit log hatası: {e}")
                flash("Giriş başarılı.", "success")
                return redirect(url_for(".index"))
            else:
                flash("Geçersiz kullanıcı adı veya şifre.", "error")
        return self.render("admin/login.html")
    @expose("/logout")
    def logout(self):
        session.pop("admin_logged_in", None)
        return redirect(url_for(".login"))

class UploadModelView(ModelView):
    column_list = ("id", "original_name", "risk_level", "cancer_ratio", "uploaded_at", "status")
    column_searchable_list = ("original_name", "risk_level")
    column_filters = ("risk_level", "uploaded_at")
    can_create = False
    can_edit = False
    can_delete = False

admin = Admin(app, name="Histopathology Admin", template_mode="bootstrap4", index_view=SecureAdminIndexView())
admin.add_view(UploadModelView(ImageUpload, db.session, name="Görüntü Analizleri"))

# 🛡️ ROBUST YARDIMCI FONKSİYONLAR
def safe_read_image(image_path, max_retries=3):
    """
    OpenCV ile robust görüntü okuma + retry mantığı
    """
    for attempt in range(max_retries):
        try:
            if not os.path.exists(image_path):
                print(f"⚠️ Dosya bulunamadı (deneme {attempt+1}): {image_path}")
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue
            
            if os.path.getsize(image_path) == 0:
                print(f"⚠️ Dosya boş (deneme {attempt+1}): {image_path}")
                time.sleep(0.1 * (attempt + 1))
                continue
                
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️ OpenCV okuma başarısız (deneme {attempt+1}): {image_path}")
                time.sleep(0.1 * (attempt + 1))
                continue
                
            # BGR → RGB dönüşümü
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, None  # (image, error)
            
        except Exception as e:
            print(f"⚠️ safe_read_image hatası (deneme {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, f"Görüntü okunamadı: {str(e)}"
            time.sleep(0.1 * (attempt + 1))
    return None, "Görüntü okunamadı: Maksimum deneme aşıldı"

def is_likely_histopathology(image_array):
    """
    OpenCV array üzerinden H&E doku analizi
    """
    try:
        # HSV'ye çevir (OpenCV BGR kullanır, zaten RGB'ye çevirdik)
        img_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)
        
        avg_hue = np.mean(h)
        avg_sat = np.mean(s)
        sat_std = np.std(s)
        
        # H&E heuristic: Pembe/Mor tonları, orta saturation, doku varyansı
        if avg_sat > 80 or avg_hue < 100 or avg_hue > 180:
            return False, "Görsel histopatoloji dokusu ile uyumlu değil."
        if sat_std < 15:
            return False, "Görsel yeterli doku detayı içermiyor."
        return True, ""
    except Exception as e:
        return False, f"Doku analizi hatası: {str(e)}"

def log_audit(action, details=""):
    try:
        log = AuditLog(action=action, details=details, ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()

# 🌐 ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
@csrf.exempt
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Dosya bulunamadı"}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Dosya seçilmedi"}), 400
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            return jsonify({"error": "Sadece JPG/JPEG/PNG/WebP desteklenir"}), 400

        job_id = str(uuid.uuid4())
        filename = f"{job_id}{ext}"
        save_path = UPLOAD_FOLDER / filename
        
        # Dosyayı kaydet ve flush ile diske yazdır
        file.save(str(save_path))
        if hasattr(os, 'fsync'):
            with open(save_path, 'rb') as f:
                os.fsync(f.fileno())
        
        # ✅ Robust okuma
        img, error = safe_read_image(save_path)
        if error or img is None:
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({"error": error or "Görüntü okunamadı"}), 400

        # ✅ Doku kontrolü (array üzerinden)
        is_valid, reason = is_likely_histopathology(img)
        if not is_valid:
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({"error": reason}), 400

        jobs[job_id] = {
            "status": "processing", "created_at": datetime.now(), "result": None, "error": None,
            "original_name": file.filename, "file_size_kb": os.path.getsize(save_path) // 1024,
            "mime_type": file.content_type, "ip_address": request.remote_addr
        }

        def process_in_background():
            job = jobs[job_id]
            upload_record = None
            try:
                start_ms = time.time()
                from predict import predict_image_patches
                # Array'yi geçici dosya olarak kaydedip predict'e gönder (mevcut pipeline ile uyumlu)
                temp_path = BASE_DIR / "static" / "uploads" / f"temp_{job_id}{ext}"
                cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                result = predict_image_patches(temp_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                elapsed_ms = int((time.time() - start_ms) * 1000)
                job["result"] = result
                job["status"] = "completed"

                upload_record = ImageUpload(
                    filename=filename, original_name=job["original_name"],
                    file_size_kb=job["file_size_kb"], mime_type=job["mime_type"],
                    ip_address=job["ip_address"], risk_level=result["risk_level"],
                    cancer_ratio=result["cancer_ratio"], total_patches=result["total_patches"],
                    cancer_patches=result["cancer_patches"], processing_time_ms=elapsed_ms, status="completed"
                )
                db.session.add(upload_record)
                db.session.flush()
                log_audit("UPLOAD_ANALYZED", f"ID:{upload_record.id} Risk:{result['risk_level']}")
                db.session.commit()
                print(f"✅ Job {job_id} DB'ye kaydedildi!")
            except Exception as e:
                job["error"] = str(e)
                job["status"] = "failed"
                if upload_record:
                    upload_record.status = "failed"
                    db.session.commit()
                log_audit("UPLOAD_FAILED", str(e))
                print(f"❌ Job {job_id} hatası: {e}")
                traceback.print_exc()

        threading.Thread(target=process_in_background, daemon=True).start()
        return jsonify({"job_id": job_id})
    
    except Exception as e:
        print(f"❌ /upload genel hatası: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Sunucu hatası: {str(e)}"}), 500

@app.route("/status/<job_id>", methods=["GET"])
@csrf.exempt
def get_status(job_id):
    try:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job bulunamadı"}), 404
        return jsonify({"status": job["status"], "result": job["result"], "error": job["error"]})
    except Exception as e:
        return jsonify({"error": f"Status hatası: {str(e)}"}), 500

# ⚠️ HATA YÖNETİCİLERİ
@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if request.path.startswith('/upload') or request.path.startswith('/status') or request.is_json or request.headers.get('Accept') == 'application/json':
        return jsonify({"error": "Sunucu hatası. Lütfen tekrar deneyin."}), 500
    return render_template("admin/login.html", error="Bir hata oluştu."), 500

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/upload') or request.path.startswith('/status') or request.is_json:
        return jsonify({"error": "Endpoint bulunamadı"}), 404
    abort(404)

@app.errorhandler(400)
def bad_request(error):
    if request.path.startswith('/upload') or request.path.startswith('/status') or request.is_json:
        return jsonify({"error": "Geçersiz istek"}), 400
    abort(400)

# 🚀 BAŞLANGIÇ
with app.app_context():
    try:
        db.create_all()
        if not AdminUser.query.first():
            admin_user = AdminUser(username="admin")
            admin_user.set_password(os.environ.get("ADMIN_PASSWORD", "SecurePass123!"))
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Varsayılan admin kullanıcısı oluşturuldu.")
    except Exception as e:
        print(f"⚠️ DB init hatası: {e}")

jobs = {}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
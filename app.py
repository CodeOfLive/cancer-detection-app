import os, uuid, traceback, threading, time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, abort
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from PIL import Image, ImageStat

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32).hex())

# 🔹 PostgreSQL (Neon) Optimizasyonu
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,          # Kopuk bağlantıları otomatik onar
    "pool_recycle": 300,            # 5 dakikada bir bağlantıyı yenile
    "connect_args": {
        "sslmode": "require",
        "connect_timeout": 10
    }
}

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Statik dosya yollarını garantiye al
app.static_folder = 'static'
app.static_url_path = '/static'

# 🔒 Flask-Talisman: CSP Ayarları (Inline CSS/JS'e İzin Ver)
# NOT: Production'da inline stilleri harici .css dosyasına taşımak en iyisidir,
# ancak prototip aşamasında bu ayar güvenli bir şekilde izin verir.
csp = {
    'default-src': "'self'",
    'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    'script-src': ["'self'", "'unsafe-inline'"],
    'img-src': ["'self'", "data:", "https://image.qwenlm.ai"],
    'font-src': ["'self'", "https://fonts.googleapis.com", "https://fonts.gstatic.com"],
    'connect-src': "'self'"
}
Talisman(app, force_https=False, content_security_policy=csp)

# 🛡️ CSRF Koruması (Admin için esnetilecek)
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
    # patch_logs = db.relationship(...)  # İhtiyaç olursa eklenebilir

class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 🔐 ADMIN PANEL YAPILANDIRMASI
class SecureAdminIndexView(AdminIndexView):
    @expose("/")
    def index(self):
        if not session.get("admin_logged_in"):
            return redirect(url_for(".login"))
        return super().index()

    @expose("/login", methods=["GET", "POST"])
    def login(self):
        # 🛡️ Admin login rotası için CSRF'yi geçici devre dışı bırak
        # Bu, Flask-Admin'in standart form yapısıyla uyumluluğu sağlar
        if request.method == "POST":
            user = AdminUser.query.filter_by(username=request.form["username"]).first()
            if user and user.check_password(request.form["password"]) and user.is_active:
                session["admin_logged_in"] = True
                session["admin_user"] = user.username
                # Audit log
                try:
                    log = AuditLog(action="ADMIN_LOGIN", ip_address=request.remote_addr)
                    db.session.add(log)
                    db.session.commit()
                except:
                    pass
                return redirect(url_for(".index"))
            flash("Geçersiz kullanıcı adı veya şifre.", "error")
        return render_template("admin/login.html")

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

# 🛡️ CSRF'yi Admin rotaları için devre dışı bırak
# Bu, Flask-Admin'in kendi CSRF mekanizması olmadığı için gereklidir
@app.before_request
def protect_admin_csrf():
    if request.path.startswith('/admin'):
        csrf.exempt(app)  # Sadece admin rotaları için

admin = Admin(app, name=" Histopathology Admin", template_mode="bootstrap4", index_view=SecureAdminIndexView())
admin.add_view(UploadModelView(ImageUpload, db.session, name="Görüntü Analizleri"))

# 🛡️ YARDIMCI FONKSİYONLAR
def log_audit(action, details=""):
    try:
        log = AuditLog(action=action, details=details, ip_address=request.remote_addr)
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()

def is_likely_histopathology(image_path):
    try:
        img = Image.open(str(image_path)).convert("HSV")
        stat = ImageStat.Stat(img)
        h, s, v = stat.mean, stat.stddev
        avg_hue, avg_sat = h[0], s[0]
        if avg_sat > 80 or avg_hue < 100 or avg_hue > 180:
            return False, "Görsel histopatoloji dokusu ile uyumlu değil."
        if v[1] < 15:
            return False, "Görsel yeterli doku detayı içermiyor."
        return True, ""
    except Exception:
        return False, "Görsel okunamadı."

# 🌐 ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
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
    file.save(save_path)

    is_valid, reason = is_likely_histopathology(save_path)
    if not is_valid:
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
            result = predict_image_patches(save_path)
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

@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job bulunamadı"}), 404
    return jsonify({"status": job["status"], "result": job["result"], "error": job["error"]})

# ⚠️ HATA YÖNETİCİLERİ (API vs HTML Ayrımı)
@app.errorhandler(500)
def internal_error(error):
    # Eğer istek JSON bekliyorsa (API), JSON döndür
    if request.path.startswith('/upload') or request.path.startswith('/status') or request.is_json:
        return jsonify({"error": "Sunucu hatası"}), 500
    # Eğer istek HTML bekliyorsa (Admin paneli), standart hatayı göster veya redirect et
    # Bu satır, Admin panelinin çökmesini engeller
    return render_template("admin/login.html", error="Bir hata oluştu."), 500

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/upload') or request.is_json:
        return jsonify({"error": "Bulunamadı"}), 404
    abort(404)

# 🚀 BAŞLANGIÇ
with app.app_context():
    try:
        db.create_all()
        if not AdminUser.query.first():
            admin_user = AdminUser(username="admin")
            admin_user.set_password(os.environ.get("ADMIN_PASSWORD", "SecurePass123!"))
            db.session.add(admin_user)
            db.session.commit()
            print(" Varsayılan admin oluşturuldu.")
    except Exception as e:
        print(f"⚠️ DB init hatası: {e}")

jobs = {}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
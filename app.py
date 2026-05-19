import os, uuid, traceback, threading, time, json, csv, io
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, Response, abort
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman
from flask_wtf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
import cv2
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# 🔐 Session & Security Config (Render için optimize)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32).hex())
app.config["SESSION_COOKIE_SECURE"] = False  # Render proxy'si nedeniyle False (HTTPS zaten zorunlu)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)
app.config["TRUSTED_PROXIES"] = ['127.0.0.1', '::1']  # Flask için proxy ayarı

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

# 🔒 Talisman CSP (proxy_count KALDIRILDI)
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
    image_dimensions = db.Column(db.String(20))
    confidence_score = db.Column(db.Float)
    patch_details = db.Column(db.Text)
    reviewed_by = db.Column(db.String(50))

class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# 🔐 ADMIN AUTH DEKORATÖRÜ
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            print(f"⚠️ Yetkisiz erişim: {request.path}, session: {session.get('admin_logged_in')}")
            flash("Lütfen önce giriş yapın.", "error")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function

# 🛡️ YARDIMCI FONKSİYONLAR
def log_audit(action, details="", ip_address=None):
    try:
        log = AuditLog(action=action, details=details, ip_address=ip_address or "system")
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()

def safe_read_image(image_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            if not os.path.exists(image_path):
                time.sleep(0.1 * (attempt + 1))
                continue
            if os.path.getsize(image_path) == 0:
                time.sleep(0.1 * (attempt + 1))
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                time.sleep(0.1 * (attempt + 1))
                continue
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"Görüntü okunamadı: {str(e)}"
            time.sleep(0.1 * (attempt + 1))
    return None, "Görüntü okunamadı: Maksimum deneme aşıldı"

def is_likely_histopathology(image_array):
    try:
        img_hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv)
        avg_hue, avg_sat = np.mean(h), np.mean(s)
        sat_std = np.std(s)
        if avg_sat > 80 or avg_hue < 100 or avg_hue > 180:
            return False, "Görsel histopatoloji dokusu ile uyumlu değil."
        if sat_std < 15:
            return False, "Görsel yeterli doku detayı içermiyor."
        return True, ""
    except Exception as e:
        return False, f"Doku analizi hatası: {str(e)}"

# 🚀 VERİTABANI MIGRATION
def migrate_database():
    with app.app_context():
        try:
            conn = db.engine.raw_connection()
            cursor = conn.cursor()
            columns_to_add = {
                "image_dimensions": "VARCHAR(20)",
                "confidence_score": "FLOAT",
                "patch_details": "TEXT",
                "reviewed_by": "VARCHAR(50)"
            }
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'uploads'")
            existing_cols = [row[0] for row in cursor.fetchall()]
            for col_name, col_type in columns_to_add.items():
                if col_name not in existing_cols:
                    print(f"🔧 Eksik sütun ekleniyor: {col_name} {col_type}")
                    cursor.execute(f'ALTER TABLE uploads ADD COLUMN IF NOT EXISTS {col_name} {col_type}')
                    conn.commit()
            cursor.close()
            conn.close()
            print("✅ Veritabanı migration tamamlandı.")
        except Exception as e:
            print(f"⚠️ Migration hatası: {e}")

# 🌐 ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/admin/login", methods=["GET", "POST"])
@csrf.exempt
def admin_login():
    print(f"🔍 Login isteği: {request.method}, session öncesi: {session.get('admin_logged_in')}")
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        print(f"🔐 Login denemesi: username={username}")
        
        try:
            user = AdminUser.query.filter_by(username=username).first()
            expected_pass = os.environ.get("ADMIN_PASSWORD", "SecurePass123!")
            
            if user:
                print(f"✅ Kullanıcı bulundu, password check: {user.check_password(password)}")
            
            if user and user.check_password(password) and user.is_active:
                print("✅ Giriş başarılı, session oluşturuluyor...")
                session.clear()
                session["admin_logged_in"] = True
                session["admin_user"] = username
                session.permanent = True
                session.modified = True  # Session'ın kaydedilmesini zorla
                
                try:
                    log = AuditLog(action="ADMIN_LOGIN", ip_address=request.remote_addr)
                    db.session.add(log)
                    db.session.commit()
                    print("✅ Audit log kaydedildi")
                except Exception as log_err:
                    db.session.rollback()
                    print(f"⚠️ Audit log hatası: {log_err}")
                
                flash("Giriş başarılı.", "success")
                
                # Debug log
                print(f"🍪 Session cookie ayarları: Secure={app.config['SESSION_COOKIE_SECURE']}, SameSite={app.config['SESSION_COOKIE_SAMESITE']}")
                print(f"🔑 Session içeriği: {dict(session)}")
                
                # ✅ Redirect ile GET isteği gönder
                response = redirect("/dashboard", code=302)
                print("🚀 Dashboard'a yönlendiriliyor...")
                return response
            else:
                print("❌ Giriş başarısız: kullanıcı yok veya şifre yanlış")
                flash("Geçersiz kullanıcı adı veya şifre.", "error")
                
        except Exception as e:
            print(f"❌ Login exception: {e}")
            traceback.print_exc()
            db.session.rollback()
            flash("Veritabanı hatası. Lütfen tekrar deneyin.", "error")
    
    return render_template("admin/login.html")

@app.route("/admin/logout", methods=["GET", "POST"])
@admin_required
def admin_logout():
    print("🚪 Logout isteği")
    session.clear()
    flash("Çıkış yapıldı.", "success")
    return redirect("/", code=302)

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
        
        file.save(str(save_path))
        if hasattr(os, 'fsync'):
            with open(save_path, 'rb') as f: os.fsync(f.fileno())
        
        img, error = safe_read_image(save_path)
        if error or img is None:
            if os.path.exists(save_path): os.remove(save_path)
            return jsonify({"error": error or "Görüntü okunamadı"}), 400

        is_valid, reason = is_likely_histopathology(img)
        if not is_valid:
            if os.path.exists(save_path): os.remove(save_path)
            return jsonify({"error": reason}), 400

        thread_data = {
            "job_id": job_id, "filename": filename, "save_path": save_path,
            "original_name": file.filename, "file_size_kb": os.path.getsize(save_path) // 1024,
            "mime_type": file.content_type, "ip_address": request.remote_addr
        }

        jobs[job_id] = {
            "status": "processing", "created_at": datetime.now(), "result": None, "error": None,
            **{k: v for k, v in thread_data.items() if k != "save_path"}
        }

        def process_in_background(data):
            with app.app_context():
                job = jobs[data["job_id"]]
                upload_record = None
                try:
                    start_ms = time.time()
                    from predict import predict_image_patches
                    
                    temp_path = UPLOAD_FOLDER / f"temp_{data['job_id']}{Path(data['filename']).suffix}"
                    cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    result = predict_image_patches(str(temp_path))
                    if os.path.exists(temp_path): os.remove(temp_path)
                    
                    elapsed_ms = int((time.time() - start_ms) * 1000)
                    job["result"] = result
                    job["status"] = "completed"

                    h, w, _ = img.shape
                    avg_confidence = 1.0 - result["cancer_ratio"] / 100.0
                    
                    upload_record = ImageUpload(
                        filename=data["filename"], original_name=data["original_name"],
                        file_size_kb=data["file_size_kb"], mime_type=data["mime_type"],
                        ip_address=data["ip_address"], risk_level=result["risk_level"],
                        cancer_ratio=result["cancer_ratio"], total_patches=result["total_patches"],
                        cancer_patches=result["cancer_patches"], processing_time_ms=elapsed_ms, 
                        status="completed",
                        image_dimensions=f"{w}x{h}",
                        confidence_score=round(avg_confidence, 3),
                        patch_details=json.dumps({"cancer_patches": result["cancer_patches"], "total": result["total_patches"]}),
                        reviewed_by=None
                    )
                    db.session.add(upload_record)
                    db.session.flush()
                    
                    log_audit("UPLOAD_ANALYZED", f"ID:{upload_record.id} Risk:{result['risk_level']}", ip_address=data["ip_address"])
                    db.session.commit()
                    print(f"✅ Job {data['job_id']} DB'ye kaydedildi!")
                    
                except Exception as e:
                    job["error"] = str(e)
                    job["status"] = "failed"
                    if upload_record:
                        upload_record.status = "failed"
                    db.session.rollback()
                    log_audit("UPLOAD_FAILED", str(e), ip_address=data["ip_address"])
                    print(f"❌ Job {data['job_id']} hatası: {e}")
                    traceback.print_exc()

        threading.Thread(target=process_in_background, args=(thread_data,), daemon=True).start()
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

@app.route("/admin/export/csv", methods=["GET", "POST"])
@admin_required
@csrf.exempt
def export_csv():
    try:
        uploads = ImageUpload.query.all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Orijinal İsim", "Risk Seviyesi", "Kanser Oranı", "Boyut", "Güven Skoru", "Patch Detayı", "İnceleyen", "Yüklenme Tarihi", "Durum"])
        for u in uploads:
            writer.writerow([u.id, u.original_name, u.risk_level, f"%{u.cancer_ratio}", u.image_dimensions or "N/A", u.confidence_score if u.confidence_score else "N/A", u.patch_details or "N/A", u.reviewed_by or "N/A", u.uploaded_at.strftime("%d.%m.%Y %H:%M"), u.status])
        return Response(output.getvalue(), mimetype="text/csv; charset=utf-8", headers={"Content-Disposition": f"attachment;filename=analizler_{datetime.now().strftime('%Y%m%d')}.csv"})
    except Exception as e:
        print(f"❌ CSV export hatası: {e}")
        return jsonify({"error": "CSV export hatası"}), 500

@app.route("/dashboard", methods=["GET", "POST"])
@admin_required
@csrf.exempt
def dashboard():
    try:
        print(f"📊 Dashboard erişimi, session: {session.get('admin_logged_in')}")
        total = ImageUpload.query.count()
        high_risk = ImageUpload.query.filter_by(risk_level="yüksek").count()
        medium_risk = ImageUpload.query.filter_by(risk_level="orta").count()
        low_risk = ImageUpload.query.filter_by(risk_level="düşük").count()
        today = ImageUpload.query.filter(db.func.date(ImageUpload.uploaded_at) == datetime.utcnow().date()).count()
        recent = ImageUpload.query.order_by(ImageUpload.uploaded_at.desc()).limit(10).all()
        return render_template("dashboard.html", stats={"total": total, "high_risk": high_risk, "medium_risk": medium_risk, "low_risk": low_risk, "today": today}, recent=recent)
    except Exception as e:
        print(f"❌ Dashboard hatası: {e}")
        return render_template("admin/login.html", error="Dashboard yüklenemedi."), 500

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
    return abort(404)

@app.errorhandler(405)
def method_not_allowed(error):
    if request.path.startswith('/upload') or request.path.startswith('/status') or request.is_json:
        return jsonify({"error": "Bu endpoint sadece POST/GET destekler"}), 405
    flash("Yöntem desteklenmiyor.", "error")
    return redirect(url_for("home"))

@app.errorhandler(400)
def bad_request(error):
    if request.path.startswith('/upload') or request.path.startswith('/status') or request.is_json:
        return jsonify({"error": "Geçersiz istek"}), 400
    return abort(400)

# 🚀 BAŞLANGIÇ
if __name__ == "__main__":
    migrate_database()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "cancer_classifier.h5"
IMG_SIZE = 224
PATCH_SIZE = 224

_model = None

def get_model():
    global _model
    if _model is None:
        print("🔄 Model yükleniyor...")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("✅ Model hazır!")
    return _model

def predict_image_patches(image_path, threshold=0.8, max_dimension=400):
    model = get_model()
    start_time = time.time()

    img = cv2.imread(str(image_path))
    if img is None: raise ValueError("Görüntü okunamadı")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        
    if h < PATCH_SIZE or w < PATCH_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        h, w = IMG_SIZE, IMG_SIZE

    cancer_count = 0
    total_patches = 0
    
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            if time.time() - start_time > 25: break
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = patch.astype("float32") / 255.0
            patch = np.expand_dims(patch, axis=0)
            pred = model.predict(patch, verbose=0)[0][0]
            if (1.0 - pred) > threshold: cancer_count += 1
            total_patches += 1
        if time.time() - start_time > 25: break

    ratio = (cancer_count / total_patches * 100) if total_patches > 0 else 0
    risk = "düşük" if ratio < 30 else ("orta" if ratio < 70 else "yüksek")
    
    print(f"✅ {time.time()-start_time:.1f}s | {total_patches} patch | %{ratio} {risk}")
    return {
        "cancer_ratio": round(ratio, 2), "risk_level": risk,
        "total_patches": total_patches, "cancer_patches": cancer_count,
        "warning": "⚠️ Bu sistem tıbbi teşhis koymaz, sadece tahmini risk analizi yapar."
    }
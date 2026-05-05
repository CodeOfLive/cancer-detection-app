import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "cancer_classifier.keras"

print("🔄 Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 224
PATCH_SIZE = 224
print("✅ Model hazır!")

def predict_image_patches(image_path, threshold=0.8):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    patch_h, patch_w = PATCH_SIZE, PATCH_SIZE
    
    cancer_count = 0
    total_patches = 0
    
    for y in range(0, h - patch_h + 1, patch_h):
        for x in range(0, w - patch_w + 1, patch_w):
            patch = img[y:y+patch_h, x:x+patch_w]
            patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
            
            # Model ilk katmanda Rescaling(1./255) yapıyor, biz sadece float32 çeviriyoruz
            patch = patch.astype("float32")
            patch = np.expand_dims(patch, axis=0)
            
            pred = model.predict(patch, verbose=0)[0][0]
            
            # 🔧 DÜZELTME: Model 0=Kanser, 1=Normal öğrendi.
            # Bu yüzden kanser olasılığını tersine çeviriyoruz.
            cancer_prob = 1.0 - pred
            
            if cancer_prob > threshold:
                cancer_count += 1
            total_patches += 1
    
    cancer_ratio = (cancer_count / total_patches * 100) if total_patches > 0 else 0
    
    if cancer_ratio < 30:
        risk_level = "düşük"
    elif cancer_ratio < 70:
        risk_level = "orta"
    else:
        risk_level = "yüksek"
    
    return {
        "cancer_ratio": round(cancer_ratio, 2),
        "risk_level": risk_level,
        "total_patches": total_patches,
        "cancer_patches": cancer_count,
        "warning": "️ Bu sistem tıbbi teşhis koymaz, sadece tahmini risk analizi yapar."
    }

def debug_single_prediction(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    patch = img[0:224, 0:224]
    patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
    patch = patch.astype("float32")
    patch = np.expand_dims(patch, axis=0)
    
    pred = model.predict(patch, verbose=0)[0][0]
    cancer_prob = 1.0 - pred
    
    print(f"🔍 DEBUG - Model çıktısı (raw): {pred:.4f}")
    print(f"🔍 DEBUG - Kanser Olasılığı (düzeltildi): {cancer_prob:.4f}")
    print(f"   Threshold: 0.5")
    print(f"   Sonuç: {'KANSER' if cancer_prob > 0.5 else 'NORMAL'}")
    
    return pred
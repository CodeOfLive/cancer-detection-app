import os, time, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image  # Keras utils yerine PIL kullanıyoruz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "cancer_classifier.h5"
IMG_SIZE, PATCH_SIZE = 224, 224
MAX_PATCHES = 9

_model = None

def get_model():
    global _model
    if _model is None:
        print("🔄 Model yükleniyor...")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("✅ Model hazır!")
    return _model

def predict_image_patches(image_path, threshold=0.8):
    model = get_model()
    start = time.time()
    
    try:
        # PIL ile aç & thumbnail ile yerinde küçült (RAM şişmesini önler)
        img = Image.open(str(image_path)).convert("RGB")
        img.thumbnail((672, 672), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Görüntü okunamadı veya format desteklenmiyor: {e}")

    h, w, _ = img_array.shape
    
    # Patch boyutundan küçükse 224x224'e sabitle
    if h < PATCH_SIZE or w < PATCH_SIZE:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        h, w = IMG_SIZE, IMG_SIZE

    cancer_count, total_patches = 0, 0
    
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            if total_patches >= MAX_PATCHES: break
            
            patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch = np.expand_dims(patch, axis=0)
            
            pred = model.predict(patch, verbose=0)[0][0]
            cancer_prob = 1.0 - pred
            
            if cancer_prob > threshold:
                cancer_count += 1
            total_patches += 1
            
        if total_patches >= MAX_PATCHES: break

    ratio = (cancer_count / total_patches * 100) if total_patches > 0 else 0
    risk = "düşük" if ratio < 30 else ("orta" if ratio < 70 else "yüksek")
    
    print(f"✅ {time.time()-start:.1f}s | {total_patches} patch | %{ratio} {risk}")
    
    return {
        "cancer_ratio": round(ratio, 2),
        "risk_level": risk,
        "total_patches": total_patches,
        "cancer_patches": cancer_count,
        "warning": "⚠️ Bu sistem tıbbi teşhis koymaz, sadece tahmini risk analizi yapar."
    }
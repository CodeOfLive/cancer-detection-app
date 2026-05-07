import os, time, numpy as np, tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array

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
        #  Keras Native Yükleme (RGB + Float32 + [0, 255])
        # target_size ile resmi önceden 672x672'ye küçültüyoruz (3x3 patch için)
        img = load_img(str(image_path), target_size=(672, 672))
        img_array = img_to_array(img)
    except Exception as e:
        raise ValueError(f"Görüntü yüklenemedi: {e}")

    h, w, _ = img_array.shape
    
    # Eğer görsel patch boyutundan küçükse 224x224'e sabitle
    if h < PATCH_SIZE or w < PATCH_SIZE:
        img = load_img(str(image_path), target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        h, w = IMG_SIZE, IMG_SIZE

    cancer_count, total_patches = 0, 0
    
    # Patch'lere böl
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            if total_patches >= MAX_PATCHES: break
            
            # Patch'i kes
            patch = img_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Modelin Rescaling(1./255) katmanı olduğu için [0, 255] aralığını koruyoruz.
            # Sadece float32 olduğundan emin ol.
            patch = patch.astype("float32")
            patch = np.expand_dims(patch, axis=0)
            
            # Tahmin
            pred = model.predict(patch, verbose=0)[0][0]
            
            # colon_aca=0 (Kanser), colon_n=1 (Normal) olduğu için:
            cancer_prob = 1.0 - pred
            
            if cancer_prob > threshold:
                cancer_count += 1
            total_patches += 1
            
        if total_patches >= MAX_PATCHES: break

    ratio = (cancer_count / total_patches * 100) if total_patches > 0 else 0
    risk = "düşük" if ratio < 30 else ("orta" if ratio < 70 else "yüksek")
    
    elapsed = time.time() - start
    print(f"✅ {elapsed:.1f}s | {total_patches} patch | %{ratio} {risk}")
    
    return {
        "cancer_ratio": round(ratio, 2),
        "risk_level": risk,
        "total_patches": total_patches,
        "cancer_patches": cancer_count,
        "warning": "⚠️ Bu sistem tıbbi teşhis koymaz, sadece tahmini risk analizi yapar."
    }
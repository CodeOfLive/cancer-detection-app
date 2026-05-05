import tensorflow as tf
import os
import matplotlib.pyplot as plt
from pathlib import Path

# 📁 Yollar
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "train"
MODEL_SAVE_PATH = BASE_DIR / "models" / "cancer_classifier.keras"
os.makedirs("models", exist_ok=True)

# ⚙️ Hyperparametreler
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

print("🚀 Veri pipeline'ı oluşturuluyor...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    seed=SEED, validation_split=0.2, subset="training", label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    seed=SEED, validation_split=0.2, subset="validation", label_mode="binary"
)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 🌪️ Data Augmentation (Overfitting engelleme)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# 🧠 MobileNetV2 Transfer Learning
print("🧠 MobileNetV2 yükleniyor (ImageNet ağırlıkları)...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # İlk aşamada dondurulur

# 🔗 Model Mimarisi
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

# ⚙️ Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 📈 Callback'ler
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_SAVE_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

print(f"📊 Eğitim başlıyor... (Max epoch: {EPOCHS})")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# 💾 Kaydet
model.save(MODEL_SAVE_PATH)
print(f"✅ Model başarıyla kaydedildi: {MODEL_SAVE_PATH}")

# 📉 Grafikler
acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
loss, val_loss = history.history["loss"], history.history["val_loss"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy"); plt.plot(val_acc, label="Val Accuracy")
plt.title("Doğruluk (Accuracy)"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss"); plt.plot(val_loss, label="Val Loss")
plt.title("Kayıp (Loss)"); plt.legend()
plt.tight_layout()
plt.show()

print("🎉 Gün 2 tamamlandı! Model Flask API'ye bağlanmaya hazır.")
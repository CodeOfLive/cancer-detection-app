import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "train"

print(f"🔍 Kontrol edilen yol: {DATA_PATH}")

if not DATA_PATH.exists():
    print("❌ data/train klasörü bulunamadı!")
else:
    print("✅ Klasör bulundu! İçerik sayılıyor...")
    for cls in sorted(os.listdir(DATA_PATH)):
        full = DATA_PATH / cls
        if full.is_dir():
            # Hem .jpeg hem .jpg uzantılarını kabul edecek şekilde güncellendi
            count = len([f for f in os.listdir(full) if f.lower().endswith((".jpeg", ".jpg"))])
            print(f"✅ {cls}: {count} görsel")

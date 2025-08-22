import os
import json
import shutil
import hashlib
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\yas\tesseract.exe'


# ------------------------- CONFIGURATION -------------------------
INPUT_JSON = Path("output/data.json")
OUTPUT_JSON = Path("output/data_cleaned.json")
IMG_BASE_DIR = Path("output/images")
CLEAN_IMG_DIR = Path("output/images_cleaned")
REMOVED_IMG_DIR = Path("output/images_removed")
MIN_W, MIN_H = 150, 150
MAX_IMAGES_PER_PAGE = 4
MAX_TEXT_WORDS = 30  # Seuil pour filtre OCR
# -----------------------------------------------------------------

def is_large_enough(img_path):
    try:
        with Image.open(img_path) as img:
            return img.width >= MIN_W and img.height >= MIN_H
    except Exception as e:
        print(f"❌ Erreur ouverture image {img_path}: {e}")
        return False

def compute_image_hash(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB").resize((64, 64))
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception as e:
        print(f"❌ Erreur hash image {img_path}: {e}")
        return None

def contains_too_much_text(img_path):
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        words = text.strip().split()
        return len(words) > MAX_TEXT_WORDS
    except Exception as e:
        print(f"❌ Erreur OCR image {img_path}: {e}")
        return False

def move_to_removed(img_path, reason):
    rel_subdir = img_path.parent.relative_to(IMG_BASE_DIR)
    out_dir = REMOVED_IMG_DIR / reason / rel_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / img_path.name
    try:
        shutil.copy2(img_path, out_path)
    except Exception as e:
        print(f"Erreur déplacement image rejetée : {e}")

# ------------------------ INITIALISATION -------------------------

# Nettoyer anciens dossiers
for folder in [CLEAN_IMG_DIR, REMOVED_IMG_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

# Charger les données
with open(INPUT_JSON, encoding="utf-8") as f:
    data = json.load(f)

new_data = []
seen_hashes = set()

print("🚀 Filtrage des images en cours...")

# ------------------------- TRAITEMENT ----------------------------

for page in tqdm(data, desc="📄 Pages"):
    new_images = []
    img_paths = page.get("images", [])

    img_infos = []
    for img_rel_path in img_paths:
        img_rel_path_unix = img_rel_path.replace("\\", "/")
        if img_rel_path_unix.startswith("images/"):
            relative_path = img_rel_path_unix[len("images/"):]
        else:
            relative_path = img_rel_path_unix

        full_path = IMG_BASE_DIR / relative_path

        if not full_path.exists():
            print(f"⚠️ Image non trouvée : {full_path}")
            continue

        # 🔴 Trop petite
        if not is_large_enough(full_path):
            move_to_removed(full_path, "small")
            continue

        # 🔴 Trop textuelle
        if contains_too_much_text(full_path):
            move_to_removed(full_path, "textual")
            continue

        # 🔴 Doublon
        img_hash = compute_image_hash(full_path)
        if not img_hash:
            continue
        if img_hash in seen_hashes:
            move_to_removed(full_path, "duplicate")
            continue

        seen_hashes.add(img_hash)
        img_infos.append((full_path, img_hash))

    # 🔴 Garde max 4 images par page
    img_infos = img_infos[:MAX_IMAGES_PER_PAGE]

    for full_path, _ in img_infos:
        rel_subdir = full_path.parent.relative_to(IMG_BASE_DIR)
        out_dir = CLEAN_IMG_DIR / rel_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / full_path.name
        shutil.copy2(full_path, out_path)
        new_images.append(str(out_path.relative_to("output")))

    page["images"] = new_images
    new_data.append(page)

# Sauvegarde du nouveau JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

# ------------------------- FIN -------------------------

print(f"\n✅ Filtrage terminé : {len(seen_hashes)} images uniques conservées.")
print(f"📁 Images utiles : {CLEAN_IMG_DIR}")
print(f"📁 Images supprimées : {REMOVED_IMG_DIR}")
print(f"📝 Fichier JSON mis à jour : {OUTPUT_JSON}")

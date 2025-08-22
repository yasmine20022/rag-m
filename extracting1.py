
import json
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm
import re

# === CONFIGURATION ===
INPUT_DIR   = Path("input_pdfs")      # Dossier où tu déposes tes PDF
OUTPUT_DIR  = Path("output")          # Dossier global de sortie
MIN_IMAGE   = 100                       # Taille minimale en px pour extraire l'image
RASTER_DPI  = 300                       # DPI pour rasterisation des pages
JSON_PATH   = OUTPUT_DIR / "data.json"
TXT_PATH    = OUTPUT_DIR / "full_text.txt"

# ------------------------------------------------------------------ #
def slugify(name: str) -> str:
    """Renvoie un identifiant sûr basé sur le nom du PDF (sans extension)."""
    return re.sub(r"[^\w\-]", "_", name.lower())

# ------------------------------------------------------------------ #

def save_binary(data: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# ------------------------------------------------------------------ #

def extract_one_pdf(pdf_path: Path, pdf_id: str):
    """Extrait texte, images et raster pour un PDF, retourne dicts."""
    # Texte par page
    text_pages = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = (page.extract_text() or "").strip()
            if content:
                text_pages[page.page_number] = content

    # Images et raster
    doc = fitz.open(pdf_path)
    img_dir    = OUTPUT_DIR / "images" / pdf_id
    raster_dir = OUTPUT_DIR / "rasterized_pages" / pdf_id
    imgs_rast  = {}

    for idx in range(len(doc)):
        page_num = idx + 1
        page = doc[idx]
        entry = {"images": [], "raster": None}

        # Images intégrées
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            w, h = base.get("width", 0), base.get("height", 0)
            if w < MIN_IMAGE or h < MIN_IMAGE:
                continue

            ext = base['ext'].lower()
            ext = 'png' if ext not in ('png', 'jpg', 'jpeg') else ext
            fname = f"{pdf_id}_p{page_num:03d}_img{img_idx+1}.{ext}"
            fpath = img_dir / fname
            save_binary(base['image'], fpath)
            entry['images'].append(str(fpath.relative_to(OUTPUT_DIR)))

        # Rasterisation
        pix = page.get_pixmap(dpi=RASTER_DPI)
        rfname = f"{pdf_id}_p{page_num:03d}_raster.png"
        rpath = raster_dir / rfname
        save_binary(pix.tobytes("png"), rpath)
        entry['raster'] = str(rpath.relative_to(OUTPUT_DIR))

        imgs_rast[page_num] = entry

    doc.close()
    return text_pages, imgs_rast

# ===================== PIPELINE GLOBAL ============================ #

def main():
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {INPUT_DIR.resolve()}")

    # Préparer dossiers de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rasterized_pages").mkdir(parents=True, exist_ok=True)

    all_json = []
    all_text = []

    print(f"➡️ Traitement de {len(pdf_files)} PDF...")
    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        pdf_id = slugify(pdf_path.stem)
        (OUTPUT_DIR / "images" / pdf_id).mkdir(exist_ok=True)
        (OUTPUT_DIR / "rasterized_pages" / pdf_id).mkdir(exist_ok=True)

        text_pages, ir_pages = extract_one_pdf(pdf_path, pdf_id)

        pages = sorted(set(text_pages) | set(ir_pages))
        for p in pages:
            entry_json = {
                'pdf': pdf_path.name,
                'pdf_id': pdf_id,
                'page': p,
                'text': text_pages.get(p, ""),
                'images': ir_pages.get(p, {}).get('images', []),
                'raster': ir_pages.get(p, {}).get('raster')
            }
            all_json.append(entry_json)

            all_text.append(f"\n\n===== {pdf_path.name} | Page {p} =====\n\n" + text_pages.get(p, ""))

    # Sauvegarde JSON et TXT global
    save_text(json.dumps(all_json, indent=2, ensure_ascii=False), JSON_PATH)
    save_text("".join(all_text), TXT_PATH)

    print("\n✅ Extraction batch terminée !")
    print(f"• JSON global       : {JSON_PATH.resolve()}")
    print(f"• Texte global      : {TXT_PATH.resolve()}")
    print(f"• Dossier images    : { (OUTPUT_DIR/'images').resolve() }")
    print(f"• Dossier raster    : { (OUTPUT_DIR/'rasterized_pages').resolve() }")

if __name__ == "__main__":
    main()


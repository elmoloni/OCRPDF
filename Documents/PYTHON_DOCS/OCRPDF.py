from __future__ import annotations

# Essential runtime imports required before function/type annotations
import os
import sys
import io
import re
import time
import logging
import shutil
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil as _shutil
import argparse

import fitz  # PyMuPDF
import pytesseract  # OCR
from PIL import Image
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import pikepdf  # for encryption / permissions
import cv2
import numpy as np

def _pick_tesseract_and_tessdata():
    base = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    localapp = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\%USERNAME%\AppData\Local"))

    # Candidate executables
    cand_exec = [
        base / "Tesseract-OCR" / "tesseract.exe",                           # bundled (PyInstaller)
        Path(os.environ.get("TESSERACT_PATH", "")),                         # env override
        localapp / "Programs" / "Tesseract-OCR" / "tesseract.exe",          # user-local install
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    exe = next((p for p in cand_exec if p and p.exists()), None)
    if exe is None:
        found = _shutil.which("tesseract")
        exe = Path(found) if found else None

    # Candidate tessdata dirs
    cand_tess = []
    if exe:
        cand_tess += [exe.parent / "tessdata"]                              # alongside the exe
    cand_tess += [
        base / "Tesseract-OCR" / "tessdata",                                # bundled
        base / "tessdata",                                                  # project-local
    ]
    # If user provided TESSDATA_PREFIX, consider both forms
    tdp = os.environ.get("TESSDATA_PREFIX")
    if tdp:
        cand_tess += [Path(tdp), Path(tdp) / "tessdata"]

    tess = next((p for p in cand_tess if p and p.exists()), None)
    return (str(exe) if exe else None, str(tess) if tess else None)

tesseract_cmd, tessdata_dir = _pick_tesseract_and_tessdata()

if not tesseract_cmd:
    logging.error("tesseract.exe not found. Bundle it or install Tesseract.")
else:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# # --------------------------------------------------------------------------------------
# # Logging
# # --------------------------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# --------------------------------------------------------------------------------------
# Deterministic langdetect
# --------------------------------------------------------------------------------------
DetectorFactory.seed = 0

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_DPI = 300
# Size / compression tuning (reduce output PDF size)
TARGET_DPI = 200          # render / embedded DPI for generated PDFs
JPEG_QUALITY = 70         # JPEG quality for color pages (0-100)
MAX_PIXEL_WIDTH = 2500    # downscale very wide pages to this width


def is_pdf_user_locked(pdf_path: str) -> bool:
    """
    Returns True if the PDF is encrypted and requires a user/owner password.
    We do NOT prompt; caller decides to skip or copy-through.
    """
    try:
        with pikepdf.open(pdf_path):
            return False
    except pikepdf.PasswordError:
        logging.info(f"PDF is encrypted (user-locked): {pdf_path}")
        return True
    except Exception as e:
        logging.error(f"Error checking encryption for {pdf_path}: {e}")
        # Be conservative: treat as locked so we don't break
        return True



def _get_pdf_root(pdf) -> dict:
    """Return PDF catalog dictionary across pikepdf versions.

    Newer pikepdf exposes `pdf.root`; older versions require accessing
    `pdf.trailer['/Root']`. This helper normalizes both forms.
    """
    try:
        return pdf.root  # newer pikepdf
    except AttributeError:
        return pdf.trailer['/Root']  # older pikepdf


def has_restrictions_no_password(pdf_path: str) -> bool:
    """
    Returns True if the PDF imposes restrictions via catalog/DocMDP
    (without encryption). If the file is encrypted, returns False (we skip).
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            root = _get_pdf_root(pdf)

            # Adobe Usage Rights / Permissions catalog entry
            if '/Perms' in root:
                return True

            # DocMDP certification via signature reference
            af = root.get('/AcroForm', None)
            if not af:
                return False
            fields = af.get('/Fields', pikepdf.Array())
            for fld in fields:
                try:
                    sig = fld.get('/V', None) or fld.get('/DV', None)
                    if not sig:
                        continue
                    refs = sig.get('/Reference', None)
                    if not refs:
                        continue
                    for ref in refs:
                        try:
                            tm = ref.get('/TransformMethod', None)
                        except Exception:
                            tm = None
                        # compare either Name object or string form
                        if tm == pikepdf.Name('/DocMDP') or str(tm) == '/DocMDP':
                            return True
                except Exception:
                    continue
            return False
    except pikepdf.PasswordError:
        # user-locked: we leave it as-is, so "no removable restrictions"
        logging.info(f"Encrypted (user-locked): {pdf_path}")
        return False
    except Exception as e:
        logging.warning(f"Restriction detection failed for {pdf_path}: {e}")
        return False


def strip_restrictions_inplace(pdf_path: str) -> bool:
    """
    Removes /Perms and DocMDP references. Does NOT prompt for passwords.
    Saves back unencrypted to the same path. Returns True if a rewrite occurred.
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            root = _get_pdf_root(pdf)
            changed = False

            # /Perms entfernen (Adobe Usage Rights / Zertifikats-Policies)
            if '/Perms' in root:
                try:
                    del root['/Perms']
                    changed = True
                    logging.info(f"Removed /Perms in: {os.path.basename(pdf_path)}")
                except Exception as e:
                    logging.warning(f"Could not remove /Perms: {e}")

            # DocMDP-References in Signaturen entfernen (kann Signaturen ungültig machen)
            af = root.get('/AcroForm', None)
            if af:
                fields = af.get('/Fields', pikepdf.Array())
                for fld in list(fields):
                    try:
                        sig = fld.get('/V', None) or fld.get('/DV', None)
                        if not sig or '/Reference' not in sig:
                            continue
                        oldrefs = sig['/Reference']
                        newrefs = pikepdf.Array(
                            r for r in oldrefs
                            if r.get('/TransformMethod', None) != pikepdf.Name('/DocMDP')
                        )
                        if len(newrefs) != len(oldrefs):
                            sig['/Reference'] = newrefs
                            changed = True
                    except Exception:
                        continue

            if changed:
                # unverschlüsselt speichern; kein Linearize -> sparsamer
                pdf.save(pdf_path, encryption=None)
                return True
            return False
    except pikepdf.PasswordError:
        # user-locked: skip
        return False
    except Exception as e:
        logging.error(f"Failed to strip restrictions from {pdf_path}: {e}")
        return False




def is_page_machine_readable(page: fitz.Page) -> bool:
    """Heuristic: page has extractable text."""
    try:
        return bool(page.get_text().strip())
    except Exception:
        return False


def pixmap_to_pil(pixmap: fitz.Pixmap) -> Image.Image:
    # Convert Pixmap samples directly to a PIL Image without PNG roundtrip
    try:
        n = getattr(pixmap, "n", None)
        w = getattr(pixmap, "width", None) or getattr(pixmap, "w")
        h = getattr(pixmap, "height", None) or getattr(pixmap, "h")
        samples = getattr(pixmap, "samples", None)
        if samples is None or w is None or h is None:
            # Fallback to safe PNG route if attributes missing
            img_bytes = pixmap.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_bytes))
            return pil_image.convert("RGB")

        if n == 1:
            mode = "L"
        elif n == 4:
            mode = "RGBA"
        else:
            mode = "RGB"

        # Create image directly from raw bytes
        pil_image = Image.frombytes(mode, (w, h), samples)
        return pil_image.convert("RGB")
    except Exception:
        # Last-resort: use existing PNG fallback
        img_bytes = pixmap.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_bytes))
        return pil_image.convert("RGB")


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """Simple grayscale + Otsu threshold."""
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    _, processed_image = cv2.threshold(
        open_cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return Image.fromarray(processed_image)


def detect_language(text: str) -> str:
    """Map langdetect code to tessdata code."""
    try:
        detected_lang = detect(text)
        lang_code_map = {
            "en": "eng",
            "de": "deu",
            "it": "ita",
            "es": "spa",
            "fr": "fra",
        }
        return lang_code_map.get(detected_lang, "eng")
    except LangDetectException:
        return "eng"


def detect_lang_for_pdf(input_pdf_path: str) -> str:
    """Detect language from first 1–3 pages with text; fallback to deu+eng."""
    try:
        with fitz.open(input_pdf_path) as doc:
            texts = []
            for i in range(min(3, doc.page_count)):
                t = doc.load_page(i).get_text().strip()
                if t:
                    texts.append(t)
            if texts:
                detected = detect(" ".join(texts))
                if detected in {"de", "en"}:
                    return {"de": "deu", "en": "eng"}[detected]
                return detect_language(" ".join(texts))
    except Exception as e:
        logging.warning(f"Language detection fallback for {input_pdf_path}: {e}")
    # Fallback combo (ensure both exist)
    parts = ["deu", "eng"]
    missing = [p for p in parts if not os.path.exists(os.path.join(tessdata_dir, f"{p}.traineddata"))]
    return "eng" if missing else "deu+eng"


def is_page_black_and_white(pil_image: Image.Image) -> bool:
    """Heuristic: return True if the page is effectively black-and-white.

    Strategy:
    - Compute mean saturation in HSV; low saturation implies grayscale/BW.
    - If saturation ambiguous, downsample and count unique colors; very few colors -> BW.
    """
    try:
        arr = np.array(pil_image.convert('RGB'))
        if arr.size == 0:
            return False
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        if mean_sat < 15:  # tuned threshold
            return True

        # downsample to speed up unique color count
        height, width = arr.shape[:2]
        target_w = 200
        target_h = max(1, int(target_w * (height / width)))
        small = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3)
        uniq = np.unique(pixels, axis=0)
        if uniq.shape[0] < 512:
            return True
    except Exception:
        return False
    return False


def ocr_image(pil: Image.Image, lang: str) -> Optional[fitz.Document]:
    """
    OCR a PIL image using Tesseract's built-in PDF generator.
    Returns a one-page PDF (fitz.Document) that contains the raster image
    and an invisible searchable text layer.
    """
    try:
        # Resize very large images to reduce embedded size
        try:
            if pil.width > MAX_PIXEL_WIDTH:
                new_h = int(pil.height * (MAX_PIXEL_WIDTH / pil.width))
                pil = pil.resize((MAX_PIXEL_WIDTH, new_h), Image.LANCZOS)

            # Decide whether to binarize and which format to embed
            if is_page_black_and_white(pil):
                pil_proc = preprocess_image(pil)
                # Save as Group4 TIFF to keep B/W tiny in size
                buf = io.BytesIO()
                pil_proc.save(buf, format='TIFF', compression='group4')
                buf.seek(0)
                pil_for_tesseract = Image.open(buf).convert('RGB')
            else:
                buf = io.BytesIO()
                pil.save(buf, format='JPEG', quality=JPEG_QUALITY, optimize=True, dpi=(TARGET_DPI, TARGET_DPI))
                buf.seek(0)
                pil_for_tesseract = Image.open(buf).convert('RGB')
        except Exception as e:
            logging.debug(f"Image preflight failed, falling back to original image: {e}")
            pil_for_tesseract = pil

        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            pil_for_tesseract, lang=lang, extension='pdf', config=f'--dpi {TARGET_DPI}'
        )

        ocr_doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        try:
            chars = len(ocr_doc[0].get_text())
            logging.info(f"OCR image: chars in Tesseract PDF = {chars}")
        except Exception:
            pass
        return ocr_doc
    except Exception as e:
        logging.error(f"Error OCR'ing image: {e}")
        return None


def ocr_image_bytes(img_bytes: bytes, lang: str) -> Optional[bytes]:
    """Worker-friendly OCR: accept image bytes, return PDF bytes or None."""
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Downscale if too wide
        if pil.width > MAX_PIXEL_WIDTH:
            new_h = int(pil.height * (MAX_PIXEL_WIDTH / pil.width))
            pil = pil.resize((MAX_PIXEL_WIDTH, new_h), Image.LANCZOS)
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            pil, lang=lang, extension='pdf', config=f'--dpi {TARGET_DPI}'
        )
        return pdf_bytes
    except Exception as e:
        logging.error(f"Worker OCR failed: {e}")
        return None


def ocr_page(pdf_path: str, page_number: int, lang: str) -> Optional[fitz.Document]:
    # Backwards-compatible wrapper: render and OCR using ocr_image
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_number)
            pix = page.get_pixmap(dpi=DEFAULT_DPI)
            pil = pixmap_to_pil(pix)
            # Free pixmap memory early
            try:
                pix = None
            except Exception:
                pass
            return ocr_image(pil, lang)
    except Exception as e:
        logging.error(f"Error OCR'ing page {page_number} of {pdf_path}: {e}")
        return None


def create_searchable_pdf(input_pdf_path: str, output_pdf_path: str) -> None:
    """
    Create a searchable PDF with per-page OCR only where needed.
    - If PDF is user-locked (encrypted), copy through unchanged (no prompts).
    - After saving, remove catalog/DocMDP restrictions from the OUTPUT only if present.
    """
    try:
        # Skip user-locked files (no prompt)
        if is_pdf_user_locked(input_pdf_path):
            shutil.copyfile(input_pdf_path, output_pdf_path)
            logging.info(f"Copied user-locked PDF without changes: {output_pdf_path}")
            return

        # Language selection based on readable text
        lang = detect_lang_for_pdf(input_pdf_path)
        if "+" in lang:
            ok = all(os.path.exists(os.path.join(tessdata_dir, f"{p}.traineddata")) for p in lang.split("+"))
            use_lang = lang if ok else "eng"
        else:
            use_lang = lang if os.path.exists(os.path.join(tessdata_dir, f"{lang}.traineddata")) else "eng"

        # Build output (OCR only image pages)
        with fitz.open(input_pdf_path) as src, fitz.open() as out:
            # First pass: determine which pages need OCR and render images for them
            pages_to_ocr = []  # list of (page_index, image_bytes)
            page_actions = {}
            for i in range(src.page_count):
                page = src.load_page(i)
                if is_page_machine_readable(page):
                    page_actions[i] = 'copy'
                else:
                    page_actions[i] = 'ocr'
                    try:
                        pix = page.get_pixmap(dpi=DEFAULT_DPI)
                        pil = pixmap_to_pil(pix)
                        try:
                            pix = None
                        except Exception:
                            pass
                        buf = io.BytesIO()
                        pil.save(buf, format='JPEG', quality=85, optimize=True, dpi=(DEFAULT_DPI, DEFAULT_DPI))
                        img_bytes = buf.getvalue()
                        pages_to_ocr.append((i, img_bytes))
                    except Exception as e:
                        logging.error(f"Failed to render page {i} for OCR: {e}")
                        page_actions[i] = 'copy'

            # OCR pages in parallel using processes (reduce GIL contention for preprocessing)
            results = {}
            if pages_to_ocr:
                proc_workers = max(1, (os.cpu_count() or 2) - 1)
                with ProcessPoolExecutor(max_workers=proc_workers) as pexec:
                    future_to_index = {
                        pexec.submit(ocr_image_bytes, img_bytes, use_lang): idx
                        for idx, img_bytes in pages_to_ocr
                    }
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            pdf_bytes = future.result()
                            results[idx] = pdf_bytes
                        except Exception as e:
                            logging.error(f"Exception in process OCR for page {idx}: {e}")
                            results[idx] = None

            # Insert pages in order, using OCR results when available
            for i in range(src.page_count):
                if page_actions.get(i) == 'copy':
                    out.insert_pdf(src, from_page=i, to_page=i)
                else:
                    pdf_bytes = results.get(i)
                    if pdf_bytes:
                        try:
                            ocrd = fitz.open(stream=pdf_bytes, filetype='pdf')
                            out.insert_pdf(ocrd)
                            ocrd.close()
                        except Exception as e:
                            logging.error(f"Failed inserting OCR page {i}: {e}")
                            out.insert_pdf(src, from_page=i, to_page=i)
                    else:
                        out.insert_pdf(src, from_page=i, to_page=i)

            out.save(output_pdf_path)
            logging.info(f"Saved searchable PDF: {output_pdf_path}")

        # Conditionally strip restrictions on the OUTPUT only if present
        if has_restrictions_no_password(output_pdf_path):
            rewritten = strip_restrictions_inplace(output_pdf_path)
            logging.info(
                f"Restriction cleanup on output {'performed' if rewritten else 'not performed'}: {output_pdf_path}"
            )

        # Final verification: count extractable characters
        try:
            with fitz.open(output_pdf_path) as chk:
                total_chars = sum(len(p.get_text()) for p in chk)
                logging.info(
                    f"Verification {os.path.basename(output_pdf_path)}: "
                    f"total extractable chars = {total_chars}"
                )
        except Exception as ve:
            logging.warning(f"Verification step failed: {ve}")

    except Exception as e:
        logging.error(f"Error creating searchable PDF for {input_pdf_path}: {e}")


def sanitize_filename(filepath: str, max_length: int = 250) -> str:
    """Prefix date if missing, avoid .pdf.pdf and enforce max path length.
    
    Returns None if filename cannot be shortened to fit max_length.
    """
    directory, filename = os.path.split(filepath)

    # Remove double .pdf endings
    base, ext = os.path.splitext(filename)
    if ext.lower() == ".pdf" and base.lower().endswith(".pdf"):
        base = base[:-4]
        filename = base + ext

    # Date prefix logic
    date_prefix = datetime.now().strftime("%Y %m %d - ")
    date_patterns = [
        r"^\d{4} \d{2} \d{2}",  # YYYY MM DD
        r"^\d{4}\d{2}\d{2}",    # YYYYMMDD
        r"^\d{2}\d{2}\d{2}",    # YYMMDD
        r"^\d{2} \d{2} \d{4}",  # DD MM YYYY
        r"^\d{2} \d{2} \d{4}",  # MM DD YYYY
        r"^\d{2} \d{4}",        # MM YYYY
    ]
    if not any(re.match(pattern, base) for pattern in date_patterns):
        filename = date_prefix + filename
        base, ext = os.path.splitext(filename)

    # Enforce path length
    full_path = os.path.join(directory, filename)
    if len(full_path) > max_length:
        excess = len(full_path) - max_length
        if excess >= len(base):
            # Fallback: try a numbered filename (e.g., "1.pdf", "2.pdf", etc.)
            counter = 1
            while counter <= 999999:
                numbered_filename = f"{counter}{ext}"
                test_path = os.path.join(directory, numbered_filename)
                if len(test_path) <= max_length:
                    return test_path
                counter += 1
            # If even numbered filenames don't work, return None to skip the file
            logging.warning(f"Cannot sanitize filename within max_length {max_length}: {filepath}")
            return None
        shortened_base = base[:-excess]
        filename = shortened_base + ext

    return os.path.join(directory, filename)


def extract_date_from_filename(filename: str):
    patterns = [
        r"^\d{4} \d{2} \d{2}",  # YYYY MM DD
        r"^\d{4}\d{2}\d{2}",    # YYYYMMDD
        r"^\d{2}\d{2}\d{2}",    # YYMMDD
        r"^\d{2} \d{2} \d{4}",  # DD MM YYYY
        r"^\d{2} \d{2} \d{4}",  # MM DD YYYY
        r"^\d{2} \d{4}",        # MM YYYY
    ]
    for pattern in patterns:
        m = re.match(pattern, filename)
        if not m:
            continue
        date_str = m.group()
        try:
            if " " in date_str:
                # try most specific first
                for fmt in ("%Y %m %d", "%d %m %Y", "%m %d %Y", "%m %Y"):
                    try:
                        return datetime.strptime(date_str, fmt).date()
                    except ValueError:
                        continue
            else:
                return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
    return None


def process_single_pdf(filepath: str, source_dir: str, target_dir: str, progress_bar, start_time: float) -> None:
    relative_path = os.path.relpath(filepath, source_dir)
    target_file = os.path.join(target_dir, os.path.splitext(relative_path)[0] + ".pdf")

    sanitized_target_file = sanitize_filename(target_file)
    os.makedirs(os.path.dirname(sanitized_target_file), exist_ok=True)

    if os.path.exists(sanitized_target_file):
        existing_filename = os.path.basename(sanitized_target_file)
        existing_date = extract_date_from_filename(existing_filename)
        # If already processed today or later, skip; otherwise overwrite
        from datetime import date as _date
        if existing_date and existing_date >= _date.today():
            if progress_bar:
                progress_bar.update(1)
            return

    try:
        create_searchable_pdf(filepath, sanitized_target_file)
    except Exception as e:
        logging.error(f"Error processing {relative_path}: {e}")
        raise
    finally:
        if progress_bar:
            # tqdm provides an internal lock for thread-safety; use it if available
            lock = getattr(progress_bar, "lock", None)
            elapsed_time = time.time() - start_time
            remaining_files = max(progress_bar.total - progress_bar.n, 0)
            if lock:
                with lock:
                    progress_bar.update(1)
                    if progress_bar.n:
                        eta = (elapsed_time / progress_bar.n) * remaining_files if progress_bar.n else 0
                        progress_bar.set_postfix_str(f"ETA: {eta:.1f}s")
            else:
                progress_bar.update(1)
                if progress_bar.n:
                    eta = (elapsed_time / progress_bar.n) * remaining_files if progress_bar.n else 0
                    try:
                        progress_bar.set_postfix_str(f"ETA: {eta:.1f}s")
                    except Exception:
                        pass


def copy_non_pdfs_and_structure(source_dir: str, target_dir: str):
    non_pdf_files = []
    for root, dirs, files in os.walk(source_dir):
        # create dirs
        for d in dirs:
            target_dir_path = os.path.join(target_dir, os.path.relpath(os.path.join(root, d), source_dir))
            os.makedirs(target_dir_path, exist_ok=True)

        # copy non-PDFs
        for file in files:
            if not file.lower().endswith(".pdf"):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_dir, os.path.relpath(source_file_path, source_dir))
                sanitized_target_file = sanitize_filename(target_file_path)
                if sanitized_target_file is None:
                    # Skip files that cannot be sanitized
                    logging.warning(f"Skipping file due to path length constraints: {source_file_path}")
                    non_pdf_files.append(os.path.relpath(source_file_path, source_dir))
                    continue
                if not os.path.exists(sanitized_target_file):
                    os.makedirs(os.path.dirname(sanitized_target_file), exist_ok=True)
                    with open(source_file_path, "rb") as src, open(sanitized_target_file, "wb") as dst:
                        dst.write(src.read())
                else:
                    non_pdf_files.append(os.path.relpath(source_file_path, source_dir))
    return non_pdf_files

workers = max(4, min(10, (os.cpu_count() or 12) - 2))  # calculating based on CPU cores

def process_pdfs(source_dir: str, target_dir: str, max_workers = workers):
    os.makedirs(target_dir, exist_ok=True)

    # Copy non-PDF files and directory structure
    non_pdf_files = copy_non_pdfs_and_structure(source_dir, target_dir)

    # Collect all PDFs
    pdf_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    logging.info(f"Total PDF files found: {len(pdf_files)}")

    start_time = time.time()
    total_files = len(pdf_files)
    processed_files = 0

    try:
        # Prefer writing the progress bar to stderr so it appears in terminal windows
        terminal_isatty = (sys.stderr.isatty() or sys.stdout.isatty())
        with tqdm(
            total=total_files,
            desc="Processing PDFs",
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
            disable=not terminal_isatty,
        ) as progress_bar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pdf = {
                    executor.submit(process_single_pdf, pdf, source_dir, target_dir, progress_bar, start_time): pdf
                    for pdf in pdf_files
                }
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        future.result()
                        processed_files += 1
                    except Exception as e:
                        logging.error(f"Exception occurred while processing {pdf}: {e}")
    except Exception as e:
        logging.error(f"Error with progress bar: {e}")
        # Fallback without progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf, source_dir, target_dir, None, start_time): pdf
                for pdf in pdf_files
            }
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    future.result()
                    processed_files += 1
                    print(f"Processed {processed_files}/{total_files} files", end="\r")
                except Exception as e:
                    logging.error(f"Exception occurred while processing {pdf}: {e}")

    elapsed_time = time.time() - start_time
    logging.info(f"Processing completed in {elapsed_time:.2f} seconds")

    print(f"\nTotal files processed: {processed_files}/{total_files}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    return non_pdf_files


def summarize_process(source_dir: str, target_dir: str) -> None:
    copied_files = 0
    renamed_files = 0
    protection_removed_files = 0  # we only strip when output still has perms
    error_files = []

    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(source_file_path, source_dir)
            target_file_path = os.path.join(target_dir, relative_path)

            sanitized_target_file = sanitize_filename(target_file_path)

            if os.path.exists(sanitized_target_file):
                copied_files += 1
                if sanitized_target_file != target_file_path:
                    renamed_files += 1
                # Optional: count outputs we actually stripped
                if has_restrictions_no_password(sanitized_target_file) is False:
                    pass
            else:
                error_files.append(source_file_path)

    logging.info("Summary:")
    logging.info(f"Total files processed: {copied_files}")
    logging.info(f"Total files renamed: {renamed_files}")
    logging.info(f"Total PDF files where copy protection was removed (best effort): {protection_removed_files}")
    logging.info(f"Files with errors during process: {error_files}")


def main():
    parser = argparse.ArgumentParser(description="Batch OCR PDFs into searchable PDFs")
    parser.add_argument("--source", "-s", help="Source directory containing PDFs")
    parser.add_argument("--target", "-t", help="Explicit target directory (optional)")
    args = parser.parse_args()

    source_dir = args.source

    # If no CLI source provided, prefer console input when available
    if not source_dir:
        try:
            if sys.stdin and sys.stdin.isatty():
                source_dir = input("Enter source directory path: ").strip().strip('"')
            else:
                # No console available (frozen --noconsole); fall back to a folder picker
                root = tk.Tk()
                root.withdraw()
                source_dir = filedialog.askdirectory(title="Select source directory")
                root.destroy()
        except (KeyboardInterrupt, EOFError, RuntimeError):
            logging.error("Input aborted or not available. Exiting.")
            return

    if not source_dir:
        logging.error("No source directory provided. Exiting.")
        return

    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory does not exist: {source_dir}")
        return

    # Build automatic target directory: same parent, date prefix, original folder name + _paste
    parent = os.path.dirname(source_dir.rstrip("\\/"))
    base = os.path.basename(source_dir.rstrip("\\/"))
    date_prefix = datetime.now().strftime("%Y %m %d - ")
    target_dir = os.path.join(parent, f"{date_prefix}{base}_paste")

    # Normalize and validate
    src = os.path.abspath(source_dir)
    dst = os.path.abspath(target_dir)

    if src == dst:
        logging.error("Source and Target directories are the same. Exiting.")
        return

    common = os.path.commonpath([src, dst])
    if common == src or common == dst:
        logging.error("One directory is a subfolder of the other. Exiting.")
        return

    # Create target directory
    os.makedirs(dst, exist_ok=True)

    logging.info(f"Source: {src}")
    logging.info(f"Target: {dst}")

    process_pdfs(src, dst)
    summarize_process(src, dst)


if __name__ == "__main__":
    main()

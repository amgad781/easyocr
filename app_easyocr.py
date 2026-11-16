"""
Handwriting OCR — Streamlit + EasyOCR (100% local, offline, no API key)

Run:
    streamlit run app_easyocr.py

First run downloads the Arabic/English models (~100 MB) once, then works offline.
"""

import html
import os
import re
import statistics

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1800          # downscale large photos (keeps RAM/CPU low on free hosting)
MIN_SIDE = 1200          # upscale small images (OCR works better on larger text)
MAX_FILE_MB = 15
ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]
LANGS = ["ar", "en"]     # EasyOCR only allows Arabic to be combined with English
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


# ---------- Model (loaded once, cached) ----------
@st.cache_resource(show_spinner="Loading OCR model (first run downloads ~100 MB)…")
def load_reader():
    import easyocr
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False
    return easyocr.Reader(LANGS, gpu=use_gpu, model_storage_directory="easyocr_models", verbose=False)


# ---------- Preprocessing ----------
def preprocess_image(img: Image.Image, enhance: bool = True) -> np.ndarray:
    """Fix rotation, resize, then grayscale + CLAHE contrast + light denoise.
    Returns a numpy array EasyOCR can read directly."""
    img = ImageOps.exif_transpose(img)      # fix phone-camera rotation
    img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
    elif longest < MIN_SIDE:
        scale = MIN_SIDE / longest
    else:
        scale = 1.0
    if scale != 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img)
    if not enhance:
        return arr

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # local contrast, fixes uneven light
    gray = clahe.apply(gray)
    return gray


# ---------- OCR ----------
def group_into_lines(results):
    """EasyOCR returns boxes in arbitrary order. Group them into lines by vertical
    position, then order each line left->right (or right->left if mostly Arabic)."""
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({
            "text": text.strip(),
            "conf": float(conf),
            "x": sum(xs) / 4,
            "y": sum(ys) / 4,
            "h": max(ys) - min(ys),
        })
    if not items:
        return []

    items.sort(key=lambda i: i["y"])
    line_h = statistics.median(i["h"] for i in items) or 1
    lines, current = [], [items[0]]
    for it in items[1:]:
        if abs(it["y"] - statistics.mean(c["y"] for c in current)) < 0.6 * line_h:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)

    out = []
    for line in lines:
        joined = " ".join(i["text"] for i in line)
        arabic_chars = len(ARABIC_RE.findall(joined))
        rtl = arabic_chars > len(joined) * 0.3
        line.sort(key=lambda i: i["x"], reverse=rtl)
        out.append(" ".join(i["text"] for i in line))
    return out


def run_ocr(image_array: np.ndarray, min_conf: float):
    reader = load_reader()
    results = reader.readtext(
        image_array,
        detail=1,
        paragraph=False,
        adjust_contrast=0.5,   # helps faint pencil/pen
        contrast_ths=0.1,
        text_threshold=0.6,
        low_text=0.3,
    )
    kept = [r for r in results if r[2] >= min_conf]
    avg_conf = (sum(r[2] for r in kept) / len(kept)) if kept else 0.0
    return "\n".join(group_into_lines(kept)), avg_conf, len(results) - len(kept)


# ---------- UI ----------
st.set_page_config(page_title="Handwriting OCR (EasyOCR)", page_icon="✍️", layout="wide")
st.title("✍️ Handwriting OCR")
st.caption("Arabic + English • EasyOCR • runs 100% offline, no API key")

with st.sidebar:
    st.header("Settings")
    enhance = st.checkbox("Enhance image (grayscale + contrast + denoise)", value=True)
    min_conf = st.slider(
        "Minimum confidence", 0.0, 0.9, 0.2, 0.05,
        help="Raise to drop garbage words; lower if real words are being skipped.",
    )
    st.markdown("---")
    st.markdown(
        "**Tips:** neat, well-spaced writing works best. Dark pen on white paper, "
        "good light, no shadows, photo straight-on."
    )

uploaded = st.file_uploader("Drag & drop a handwritten image", type=ALLOWED_TYPES)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

if uploaded.size > MAX_FILE_MB * 1024 * 1024:
    st.error(f"File is too large ({uploaded.size / 1e6:.1f} MB). Max is {MAX_FILE_MB} MB.")
    st.stop()

try:
    original = Image.open(uploaded)
    original.load()
except (UnidentifiedImageError, OSError):
    st.error("Couldn't read this file as an image. Use PNG, JPG, WEBP, or BMP (HEIC isn't supported).")
    st.stop()

processed = preprocess_image(original, enhance=enhance)

left, right = st.columns(2)
with left:
    st.subheader("Image (sent to OCR)")
    st.image(processed, use_container_width=True)

with right:
    st.subheader("Extracted text")
    if st.button("Extract text", type="primary", use_container_width=True):
        try:
            with st.spinner("Reading handwriting…"):
                text, avg_conf, dropped = run_ocr(processed, min_conf)
            st.session_state["result"] = (text, avg_conf, dropped)
            st.session_state["source"] = uploaded.name
        except ImportError:
            st.error("EasyOCR isn't installed. Run: pip install easyocr")
        except (ConnectionError, OSError) as e:
            st.error(
                "Couldn't download/load the model. The first run needs internet once "
                f"(then it's cached in ~/.EasyOCR). Details: {e}"
            )
        except MemoryError:
            st.error("Out of memory. Try a smaller image.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    if st.session_state.get("result") and st.session_state.get("source") == uploaded.name:
        text, avg_conf, dropped = st.session_state["result"]
        if not text.strip():
            st.warning("No text detected. Try a clearer photo, or lower the minimum confidence.")
        else:
            st.caption(f"Average confidence: {avg_conf:.0%} • low-confidence boxes dropped: {dropped}")
            st.markdown(
                f'<div dir="auto" style="white-space:pre-wrap;font-size:1.1rem;'
                f'line-height:1.8;padding:12px;border:1px solid #8884;border-radius:8px;">'
                f"{html.escape(text)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Copy** (hover the box → copy icon) or edit below:")
            st.code(text, language=None)
            edited = st.text_area("Edit before download", value=text, height=200)
            base = os.path.splitext(uploaded.name)[0]
            st.download_button(
                "⬇️ Download .txt",
                data=edited.encode("utf-8-sig"),   # BOM so Arabic opens correctly in Notepad/Excel
                file_name=f"{base}_transcription.txt",
                mime="text/plain",
                use_container_width=True,
            )

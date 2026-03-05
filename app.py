import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imutils
import math
import pytesseract
import re

from rapidfuzz import fuzz
from detect import run_detection_pil, get_center

try:
    from Extract import extract_all_features
except ImportError as e:
    st.error(f"Error loading external module: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="Label Comparator")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp { background-color: #f8f9fa; font-family: 'Roboto', sans-serif; }
        h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm, [data-testid="stMarkdownContainer"] p strong, .stMarkdown p { color: #064b75 !important; }
        .main-header { text-align: center; color: #064b75 !important; font-weight: 700; padding-bottom: 30px; }
        div.stButton > button { background-color: #f4a303 !important; color: #ffffff !important; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; padding: 10px 24px; }
        div.stButton > button:hover { background-color: #e09600 !important; color: white !important; }
        [data-testid="stFileUploader"] { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# FILE INGESTION
# ============================================================

def pdf_to_image(uploaded_file, dpi=200):
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_page = pdf_document[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = first_page.get_pixmap(matrix=mat, alpha=False)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pdf_document.close()
    return image

def process_upload(uploaded_file, target_width=1000):
    """
    Load and width-normalise every upload to exactly 1000px wide.
    Height is preserved proportionally — this is intentional.
    Labels with different aspect ratios will have different heights
    after this step, and the alignment layer handles that correctly.
    """
    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".pdf"):
        img = pdf_to_image(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    ratio = target_width / float(img.width)
    new_height = int(float(img.height) * float(ratio))
    img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    return img

def preprocess_image(image, resize_to=None, enhance_contrast=False):
    if image is None:
        return None
    img = image.copy()
    if resize_to:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    return img

# ============================================================
# ALIGNMENT — 3-PATH ROBUST STRATEGY
# ============================================================

def _aspect_ratio(image):
    return image.width / image.height

def _ratio_differs(imageA, imageB, tolerance=0.12):
    """True if the two images have meaningfully different aspect ratios."""
    rA = _aspect_ratio(imageA)
    rB = _aspect_ratio(imageB)
    return abs(rA - rB) / max(rA, rB) > tolerance

def _scale_pad_to_base(imageB, target_w, target_h):
    """
    Scale imageB to fit within (target_w, target_h) preserving aspect ratio,
    then centre it on a white canvas of exactly that size.

    Used as:
      • Path 3 alignment fallback
      • The guaranteed safety net when alignment raises an exception
    """
    bw, bh = imageB.size
    scale = min(target_w / bw, target_h / bh)
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    resized = imageB.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas

def _try_homography(grayA, grayB, imgB_np, target_w, target_h,
                    max_features=5000, good_pct=0.15):
    """
    Full perspective homography (ORB + RANSAC).
    Best for same-ratio labels with minor perspective tilt or scan warp.
    """
    orb = cv2.ORB_create(max_features)
    kpA, desA = orb.detectAndCompute(grayA, None)
    kpB, desB = orb.detectAndCompute(grayB, None)
    if desA is None or desB is None:
        return None, False
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desA, desB)
    matches.sort(key=lambda x: x.distance)
    n = max(4, int(len(matches) * good_pct))
    matches = matches[:n]
    if len(matches) < 4:
        return None, False
    p1 = np.float32([kpA[m.queryIdx].pt for m in matches])
    p2 = np.float32([kpB[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(p2, p1, cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return None, False
    # Sanity-check: reject degenerate homographies (determinant near 0 or huge scale)
    det = np.linalg.det(H[:2, :2])
    if not (0.1 < abs(det) < 10.0):
        return None, False
    aligned = cv2.warpPerspective(imgB_np, H, (target_w, target_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
    return aligned, True

def _try_similarity(grayA, grayB, imgB_np, target_w, target_h):
    """
    Similarity transform (scale + rotation + translation, NO perspective).
    Better for different-ratio labels — keeps layout intact.
    """
    orb = cv2.ORB_create(5000)
    kpA, desA = orb.detectAndCompute(grayA, None)
    kpB, desB = orb.detectAndCompute(grayB, None)
    if desA is None or desB is None or len(desA) < 10 or len(desB) < 10:
        return None, False
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    raw = matcher.match(desA, desB)
    raw.sort(key=lambda x: x.distance)
    good = raw[:max(10, int(len(raw) * 0.15))]
    if len(good) < 6:
        return None, False
    pts_a = np.float32([kpA[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpB[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None or (inliers is not None and inliers.sum() < 4):
        return None, False
    aligned = cv2.warpAffine(imgB_np, M, (target_w, target_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return aligned, True

def align_images(imageA, imageB):
    """
    Robust 3-path alignment:

    PATH 1 — Same ratio → full homography (handles perspective/scan warp)
    PATH 2 — Different ratio → similarity transform (scale+rot+translate only,
              no perspective distortion that would destroy layout)
    PATH 3 — Fallback → scale+centre-pad (zero distortion, purely geometric)

    In all paths, the returned image is exactly imageA.size so all downstream
    pixel comparisons work without any further resizing.
    """
    try:
        target_w, target_h = imageA.size
        grayA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)

        ratio_mismatch = _ratio_differs(imageA, imageB)

        if not ratio_mismatch:
            # PATH 1: same ratio — try full homography
            grayB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2GRAY)
            imgB_np = np.array(imageB)
            aligned, ok = _try_homography(grayA, grayB, imgB_np, target_w, target_h)
            if ok:
                return Image.fromarray(aligned), True

        # PATH 2: different ratio (or homography failed) — prescale to base width
        # then attempt similarity transform
        prescale_h = int(imageB.height * (target_w / imageB.width))
        imgB_pre = imageB.resize((target_w, prescale_h), Image.Resampling.LANCZOS)
        grayB_pre = cv2.cvtColor(np.array(imgB_pre), cv2.COLOR_RGB2GRAY)
        imgB_pre_np = np.array(imgB_pre)
        aligned, ok = _try_similarity(grayA, grayB_pre, imgB_pre_np, target_w, target_h)
        if ok:
            return Image.fromarray(aligned), True

        # PATH 3: scale + centre-pad — safe, no distortion
        return _scale_pad_to_base(imageB, target_w, target_h), True

    except Exception:
        try:
            return _scale_pad_to_base(imageB, imageA.width, imageA.height), True
        except Exception:
            return imageB, False

# ============================================================
# DIFFERENCE DETECTION
# ============================================================

def find_differences(imageA, imageB, min_area=150):
    """
    SSIM pixel diff with two noise-suppression masks:

    1. Both-white mask  — suppresses white padding from scale+pad alignment
    2. Morphological close — merges nearby micro-contours to avoid
       fragmenting one real change into dozens of tiny boxes
    """
    try:
        if imageA.size != imageB.size:
            imageB = imageB.resize(imageA.size, Image.Resampling.LANCZOS)
        grayA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2GRAY)

        score, diff = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Suppress regions where both images are pure white (padding / margins)
        both_white = ((grayA > 245) & (grayB > 245)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        both_white = cv2.dilate(both_white, k, iterations=2)
        thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(both_white))

        # Merge very close diff fragments into coherent regions
        merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, merge_k)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        bboxes = []
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                x, y, w, h = cv2.boundingRect(c)
                bboxes.append((x, y, w, h))

        return {'ssim_score': score, 'bounding_boxes': bboxes, 'total_differences': len(bboxes)}
    except Exception as e:
        st.error(f"Error finding differences: {e}")
        return None

# ============================================================
# SYMBOL COMPARISON — RELATIVE-POSITION AWARE
# ============================================================

def compare_symbols(base_symbols_raw, comp_symbols_raw, base_img, comp_aligned):
    """
    Compares two symbol lists using relative position (fraction of canvas),
    so the result is invariant to:
      • Different label sizes / aspect ratios
      • White-pad offsets from scale+pad alignment
      • Minor drift from similarity-transform alignment

    Handles every scenario:
      • Symbol unchanged in same position   → silent (not reported)
      • Symbol moved to new position        → Misplaced (on child)
      • Symbol in base, missing in child    → Removed  (on base coords)
      • Symbol in child, not in base        → Added    (on child coords)
      • Multiple instances of same class    → each matched to nearest peer
        so duplicates handled correctly
      • Symbol replaced by different class  → old class = Removed, new = Added
    """
    base_w, base_h = base_img.size
    comp_w, comp_h = comp_aligned.size
    base_diag = math.sqrt(base_w ** 2 + base_h ** 2)

    # Thresholds expressed as fraction of base diagonal
    # MISPLACED_THRESH: how far a symbol must move (relative) to be called Misplaced
    #   6% of diagonal ~ 60px on a 1000px-wide typical label
    # CLAIM_RADIUS: max relative distance to consider two symbols the same instance
    #   25% of diagonal — wide enough to absorb large layout shifts without
    #   accidentally matching symbols from opposite ends of the label
    MISPLACED_THRESH = 0.06
    CLAIM_RADIUS = 0.25

    def rel_center(bbox, w, h):
        cx, cy = get_center(bbox)
        return cx / w, cy / h

    def scaled_dist(bbA, bbB):
        """Distance in 'base-pixel space' normalised by base diagonal."""
        rx_a, ry_a = rel_center(bbA, base_w, base_h)
        rx_b, ry_b = rel_center(bbB, comp_w, comp_h)
        dx = (rx_a - rx_b) * base_w
        dy = (ry_a - ry_b) * base_h
        return math.sqrt(dx * dx + dy * dy) / base_diag

    def has_content(image, bbox, dark_thresh=15):
        x1, y1, x2, y2 = map(int, bbox)
        crop = np.array(image)[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
        return np.sum(gray < 240) > dark_thresh

    claimed = set()
    result = []

    for base_sym in base_symbols_raw:
        best_idx = -1
        best_dist = float('inf')

        for idx, c_sym in enumerate(comp_symbols_raw):
            if c_sym["class"] != base_sym["class"] or idx in claimed:
                continue
            d = scaled_dist(base_sym["bbox"], c_sym["bbox"])
            if d < best_dist and d < CLAIM_RADIUS:
                best_dist = d
                best_idx = idx

        if best_idx >= 0:
            claimed.add(best_idx)
            if best_dist > MISPLACED_THRESH:
                # Symbol exists in child but shifted to a different region → Misplaced
                if has_content(comp_aligned, comp_symbols_raw[best_idx]["bbox"]):
                    box = comp_symbols_raw[best_idx].copy()
                    box["label"] = "Misplaced"
                    result.append(box)
            # else: same relative position → no change, silent
        else:
            # No matching child symbol within search radius → genuinely removed
            box = base_sym.copy()
            box["label"] = "Removed"
            result.append(box)

    # Unclaimed child symbols → new additions
    for idx, c_sym in enumerate(comp_symbols_raw):
        if idx not in claimed:
            box = c_sym.copy()
            box["label"] = "Added"
            result.append(box)

    return result

# ============================================================
# TEXT DIFF CLASSIFICATION
# ============================================================

def classify_text_boxes(ssim_boxes, base_symbols, comp_symbols,
                        base_img, comp_aligned):
    """
    Takes raw SSIM bounding boxes and classifies each as:
      Added / Deleted / Changed

    Steps:
    1. Exclude boxes that overlap with known symbol regions
       (symbols are handled separately — no double-reporting)
    2. For each remaining box, sample dark-pixel density in both
       images to determine what kind of change it is
    3. Run OCR only on confirmed diff boxes (not every SSIM fragment)
    4. Suppress OCR hallucinations: discard results that are purely
       punctuation / whitespace / fewer than 2 alphanumeric chars

    Handles:
      • Pure text change (LOT number, date)     → Changed
      • New text block (added language row)     → Added
      • Deleted text block                      → Deleted
      • Noise boxes on white padding            → suppressed by dark-pixel check
      • OCR reading barcode as garbled text     → suppressed by alphanumeric filter
    """
    all_sym_bboxes = []
    for s in list(base_symbols) + list(comp_symbols):
        x1, y1, x2, y2 = map(int, s["bbox"])
        all_sym_bboxes.append((x1, y1, x2 - x1, y2 - y1))

    def overlaps_any_symbol(box):
        bx, by, bw, bh = box
        bc = [bx, by, bx + bw, by + bh]
        for sx, sy, sw, sh in all_sym_bboxes:
            sc = [sx, sy, sx + sw, sy + sh]
            xA, yA = max(bc[0], sc[0]), max(bc[1], sc[1])
            xB, yB = min(bc[2], sc[2]), min(bc[3], sc[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter > 0:
                area_b = bw * bh
                if area_b > 0 and inter / area_b > 0.25:
                    return True
        return False

    base_gray = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2GRAY)
    child_gray = cv2.cvtColor(np.array(comp_aligned), cv2.COLOR_RGB2GRAY)
    MIN_DARK = 15

    deleted, added, changed = [], [], []

    for (x, y, w, h) in ssim_boxes:
        if overlaps_any_symbol((x, y, w, h)):
            continue
        crop_b = base_gray[y:y + h, x:x + w]
        crop_c = child_gray[y:y + h, x:x + w]
        if crop_b.size == 0 or crop_c.size == 0:
            continue
        has_b = np.sum(crop_b < 220) > MIN_DARK
        has_c = np.sum(crop_c < 220) > MIN_DARK

        if has_b and not has_c:
            deleted.append((x, y, w, h))
        elif not has_b and has_c:
            added.append((x, y, w, h))
        elif has_b and has_c:
            changed.append((x, y, w, h))

    return deleted, added, changed

def ocr_crop(image, box):
    """
    OCR a small region with 2× upscale for legibility.
    Returns empty string for results that are purely noise/punctuation.
    """
    x, y, w, h = box
    pad = 5
    img_w = image.shape[1] if isinstance(image, np.ndarray) else image.width
    img_h = image.shape[0] if isinstance(image, np.ndarray) else image.height
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    crop = np.array(image)[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray, lang='eng+fra+deu', config='--psm 6').strip()
    text = re.sub(r'[|><_~=«»"*;]', '', text).strip()
    text = re.sub(r'\n+', ' ', text)
    # Suppress hallucinations: require at least 2 alphanumeric characters
    if len(re.findall(r'[A-Za-z0-9]', text)) < 2:
        return ""
    return text

# ============================================================
# FEATURE TABLE DIFF
# ============================================================

def get_feature_diffs(base_df, comp_df, comp_type, fuzzy_threshold=85):
    if base_df.empty or comp_df.empty:
        return [], []
    base_vals = base_df[base_df['Type'] == comp_type]['Value'].tolist()
    comp_vals = comp_df[comp_df['Type'] == comp_type]['Value'].tolist()
    added, deleted = [], []

    # Barcodes and logos: exact match (they should not fuzzy-match)
    if comp_type in ['Barcode', 'Image']:
        added = list(set(comp_vals) - set(base_vals))
        deleted = list(set(base_vals) - set(comp_vals))
        return added, deleted

    # Text: fuzzy match to handle minor OCR variance
    for b in base_vals:
        if not any(fuzz.token_set_ratio(b.lower().strip(), c.lower().strip()) >= fuzzy_threshold
                   for c in comp_vals):
            deleted.append(b)
    for c in comp_vals:
        if not any(fuzz.token_set_ratio(c.lower().strip(), b.lower().strip()) >= fuzzy_threshold
                   for b in base_vals):
            added.append(c)
    return added, deleted

# ============================================================
# DRAWING
# ============================================================

def boxes_overlap(boxA, boxB, iou_threshold=0.3):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter / float(aA + aB - inter + 1e-6)
    return iou > iou_threshold

def draw_differences(image, bounding_boxes, color=(255, 0, 0), thickness=2, label=""):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in bounding_boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
        if label:
            draw.text((x, max(0, y - 15)), label, fill=color)
    return img

def draw_symbol_boxes(image, detections, color_map=None, thickness=2):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    if color_map is None:
        color_map = {"Added": (0, 255, 0), "Removed": (255, 0, 0),
                     "Misplaced": (255, 255, 0), "Symbol": (0, 0, 255)}
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d.get("label", "Symbol")
        color = color_map.get(label, (0, 0, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1, max(0, y1 - 15)), label, fill=color)
    return img

# ============================================================
# MAIN APP
# ============================================================

st.markdown('<h1 class="main-header">LABEL COMPARATOR PRO</h1>', unsafe_allow_html=True)

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    base_file = st.file_uploader("Upload Base Label", type=["jpg", "png", "jpeg", "pdf"], key="base")
with col_upload2:
    child_files = st.file_uploader("Upload Child Label(s)", type=["jpg", "png", "jpeg", "pdf"],
                                   key="child", accept_multiple_files=True)

st.write("")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    compare_clicked = st.button("Compare labels", use_container_width=True)
st.write("")

if compare_clicked:
    if base_file is not None and child_files:
        with st.status("Analyzing labels securely...", expanded=True) as status:

            # ------------------------------------------------------------------
            # BASE LABEL — process once, reuse for every child
            # ------------------------------------------------------------------
            status.write("Processing base document...")
            raw_base_img   = process_upload(base_file)
            base_processed = preprocess_image(raw_base_img, enhance_contrast=False)
            base_symbols_raw = run_detection_pil(base_processed)
            base_features_df = extract_all_features(raw_base_img, base_symbols_raw, logo_folder="logos")

            # Base symbol list for drawing (all labelled "Symbol")
            base_symbols_draw = [dict(d, label="Symbol") for d in base_symbols_raw]

            tabs = st.tabs([f.name for f in child_files])

            for tab, child_file in zip(tabs, child_files):
                with tab:
                    status.write(f"Analyzing {child_file.name}...")

                    # ----------------------------------------------------------
                    # CHILD LABEL
                    # ----------------------------------------------------------
                    raw_child_img    = process_upload(child_file)
                    comp_processed   = preprocess_image(raw_child_img, enhance_contrast=False)

                    # STEP 1 — Align child onto base canvas
                    comp_aligned, _ = align_images(base_processed, comp_processed)

                    # STEP 2 — Detect symbols on the aligned child
                    # (YOLO runs on the aligned image so symbol coordinates are
                    #  in the same pixel space as base_symbols_raw)
                    comp_symbols_raw = run_detection_pil(comp_aligned)

                    # STEP 3 — Extract full feature table from aligned child
                    comp_features_df = extract_all_features(comp_aligned, comp_symbols_raw,
                                                            logo_folder="logos")

                    # STEP 4 — SSIM pixel diff
                    diff_results = find_differences(base_processed, comp_aligned, min_area=150)
                    if not diff_results:
                        st.error(f"Error comparing '{child_file.name}'")
                        continue

                    # STEP 5 — Symbol comparison (relative-position aware)
                    comp_symbols = compare_symbols(
                        base_symbols_raw, comp_symbols_raw,
                        base_processed, comp_aligned
                    )

                    # STEP 6 — Classify text diff boxes
                    deleted_boxes, added_boxes, changed_boxes = classify_text_boxes(
                        diff_results['bounding_boxes'],
                        base_symbols_draw, comp_symbols,
                        base_processed, comp_aligned
                    )

                    # STEP 7 — OCR the diff boxes
                    added_text = [t for box in added_boxes
                                  if (t := ocr_crop(comp_aligned, box)) and len(t) > 2]
                    deleted_text = [t for box in deleted_boxes
                                    if (t := ocr_crop(base_processed, box)) and len(t) > 2]
                    modified_text = []
                    for box in changed_boxes:
                        tb = ocr_crop(base_processed, box)
                        tc = ocr_crop(comp_aligned, box)
                        if tb or tc:
                            modified_text.append(f"From: '{tb}' ➔ To: '{tc}'")

                    # STEP 8 — Feature table diffs (barcodes, logos)
                    added_bc, deleted_bc   = get_feature_diffs(base_features_df, comp_features_df, 'Barcode')
                    added_img, deleted_img = get_feature_diffs(base_features_df, comp_features_df, 'Image')

                    added_syms     = [s["class"] for s in comp_symbols if s["label"] == "Added"]
                    removed_syms   = [s["class"] for s in comp_symbols if s["label"] == "Removed"]
                    misplaced_syms = [s["class"] for s in comp_symbols if s["label"] == "Misplaced"]

                    # ----------------------------------------------------------
                    # DRAW
                    # ----------------------------------------------------------
                    base_marked = draw_differences(base_processed, deleted_boxes,
                                                   color=(255, 0, 0), label="Deleted")
                    base_marked = draw_differences(base_marked, changed_boxes,
                                                   color=(255, 165, 0), label="Changed")

                    comp_marked = draw_differences(comp_aligned, added_boxes,
                                                   color=(0, 200, 0), label="Added")
                    comp_marked = draw_differences(comp_marked, changed_boxes,
                                                   color=(255, 165, 0), label="Changed")
                    comp_marked = draw_symbol_boxes(comp_marked, comp_symbols,
                                                    color_map={"Added": (0, 200, 0),
                                                               "Removed": (255, 0, 0),
                                                               "Misplaced": (255, 200, 0)})

                    # ----------------------------------------------------------
                    # RENDER
                    # ----------------------------------------------------------
                    st.markdown("---")
                    st.markdown("### Visual Comparison")
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.markdown("**Label B (Base)**")
                        st.image(base_marked, use_container_width=True)
                    with img_col2:
                        st.markdown(f"**Label C (Child: {child_file.name})**")
                        st.image(comp_marked, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### Feature Extracted Tables")
                    feat_col1, feat_col2 = st.columns(2)
                    with feat_col1:
                        st.markdown("**Base Features**")
                        st.dataframe(base_features_df, use_container_width=True, hide_index=True)
                    with feat_col2:
                        st.markdown("**Child Features**")
                        st.dataframe(comp_features_df, use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.markdown("### 📊 Interactive Discrepancy Report")

                    diff_data = []
                    for item in added_text:    diff_data.append({"Category": "Text",    "Status": "Added",     "Value": item})
                    for item in deleted_text:  diff_data.append({"Category": "Text",    "Status": "Deleted",   "Value": item})
                    for item in modified_text: diff_data.append({"Category": "Text",    "Status": "Modified",  "Value": item})
                    for item in added_syms:    diff_data.append({"Category": "Symbol",  "Status": "Added",     "Value": item})
                    for item in misplaced_syms:diff_data.append({"Category": "Symbol",  "Status": "Misplaced", "Value": item})
                    for item in removed_syms:  diff_data.append({"Category": "Symbol",  "Status": "Deleted",   "Value": item})
                    for item in added_bc:      diff_data.append({"Category": "Barcode", "Status": "Added",     "Value": item})
                    for item in deleted_bc:    diff_data.append({"Category": "Barcode", "Status": "Deleted",   "Value": item})
                    for item in added_img:     diff_data.append({"Category": "Image",   "Status": "Added",     "Value": item})
                    for item in deleted_img:   diff_data.append({"Category": "Image",   "Status": "Deleted",   "Value": item})

                    if diff_data:
                        diff_df = pd.DataFrame(diff_data)

                        def highlight_status(row):
                            if row['Status'] == 'Added':
                                return ['background-color: rgba(40, 167, 69, 0.2); color: #155724'] * len(row)
                            elif row['Status'] == 'Deleted':
                                return ['background-color: rgba(220, 53, 69, 0.2); color: #721c24'] * len(row)
                            elif row['Status'] == 'Misplaced':
                                return ['background-color: rgba(255, 193, 7, 0.2); color: #856404'] * len(row)
                            elif row['Status'] == 'Modified':
                                return ['background-color: rgba(23, 162, 184, 0.2); color: #0c5460'] * len(row)
                            return [''] * len(row)

                        st.dataframe(diff_df.style.apply(highlight_status, axis=1),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No discrepancies found! The labels match perfectly.")

            status.update(label="Analysis Complete!", state="complete", expanded=False)

    else:
        st.error("Please ensure both the Base Label and at least one Child Label are uploaded.")

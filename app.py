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

def process_upload(uploaded_file, max_width=1000):
    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".pdf"):
        img = pdf_to_image(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int(float(img.height) * float(ratio))
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
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
# ALIGNMENT  (original)
# ============================================================

def align_images(imageA, imageB, max_features=500, good_match_percent=0.15):
    try:
        grayA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(max_features)
        keypointsA, descriptorsA = orb.detectAndCompute(grayA, None)
        keypointsB, descriptorsB = orb.detectAndCompute(grayB, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptorsA, descriptorsB)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * good_match_percent)
        matches = matches[:numGoodMatches]
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypointsA[match.queryIdx].pt
            points2[i, :] = keypointsB[match.trainIdx].pt
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        height, width = grayB.shape[:2]
        aligned = cv2.warpPerspective(np.array(imageA), h, (width, height))
        return Image.fromarray(aligned), True
    except Exception as e:
        return imageA, False

# ============================================================
# PIXEL DIFF  (original — kept for SSIM score metric only)
# ============================================================

def find_differences(imageA, imageB, threshold=0.85, min_area=150):
    try:
        if imageA.size != imageB.size:
            imageB = imageB.resize(imageA.size, Image.Resampling.LANCZOS)
        grayA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(np.array(imageB), cv2.COLOR_RGB2GRAY)
        score, diff = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        filtered_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
        bounding_boxes = [cv2.boundingRect(c) for c in filtered_cnts]
        return {'ssim_score': score, 'bounding_boxes': bounding_boxes,
                'total_differences': len(bounding_boxes)}
    except Exception as e:
        st.error(f"Error finding differences: {e}")
        return None

# ============================================================
# FEATURE-POSITION EXTRACTION  (new)
#
# Replaces the pixel-diff → OCR-crop text pipeline.
# Runs image_to_data on each label independently so results are
# immune to aspect-ratio differences and alignment drift.
# ============================================================

def get_text_lines_with_bbox(image, scale=1.5):
    """
    OCR the image and return line-level entries:
        [{'text': str, 'bbox': (x, y, w, h)}, ...]

    Words are grouped by (block_num, par_num, line_num) so multi-word
    tokens like 'TAVALLINEN KUDOSSUOJA' or 'Co. Cork, Ireland' are
    treated as one unit with a single encompassing bounding box.
    Bounding boxes are returned in original image coordinates.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray_up = cv2.resize(gray, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(
        gray_up, lang='eng+fra+deu', config='--psm 6',
        output_type=pytesseract.Output.DICT
    )
    lines = {}
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if not word:
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        if key not in lines:
            lines[key] = {'words': [], 'x1': 9999, 'y1': 9999, 'x2': 0, 'y2': 0}
        x, y, bw, bh = (data['left'][i], data['top'][i],
                        data['width'][i], data['height'][i])
        lines[key]['words'].append(word)
        lines[key]['x1'] = min(lines[key]['x1'], x)
        lines[key]['y1'] = min(lines[key]['y1'], y)
        lines[key]['x2'] = max(lines[key]['x2'], x + bw)
        lines[key]['y2'] = max(lines[key]['y2'], y + bh)
    result = []
    for v in lines.values():
        text = ' '.join(v['words']).strip()
        clean = re.sub(r'[|><_~=«»"*]', '', text).strip()
        if len(clean) > 2 and any(c.isalnum() for c in clean):
            x1 = int(v['x1'] / scale); y1 = int(v['y1'] / scale)
            x2 = int(v['x2'] / scale); y2 = int(v['y2'] / scale)
            result.append({'text': clean, 'bbox': (x1, y1, x2 - x1, y2 - y1)})
    return result


def diff_text_lines(base_lines, child_lines, fuzzy_thresh=80):
    """
    Fuzzy-compare two line-lists.
    Returns:
        deleted — lines in base missing from child  (draw red on base)
        added   — lines in child missing from base  (draw green on child)
        changed — [(base_line, child_line)] pairs   (draw orange on both)
    """
    child_texts = [l['text'] for l in child_lines]
    base_texts  = [l['text'] for l in base_lines]
    deleted, added, changed = [], [], []

    for bl in base_lines:
        best_score, best_match = 0, None
        for cl in child_lines:
            s = fuzz.token_set_ratio(bl['text'].lower(), cl['text'].lower())
            if s > best_score:
                best_score, best_match = s, cl
        if best_score < fuzzy_thresh:
            deleted.append(bl)
        elif best_score < 95 and best_match is not None:
            changed.append((bl, best_match))

    for cl in child_lines:
        if not any(fuzz.token_set_ratio(cl['text'].lower(), bt.lower()) >= fuzzy_thresh
                   for bt in base_texts):
            added.append(cl)

    return deleted, added, changed

# ============================================================
# DRAWING HELPERS  (original)
# ============================================================

def boxes_overlap(boxA, boxB, iou_threshold=0.3):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou > iou_threshold

def draw_differences(image, bounding_boxes, color=(255, 0, 0), thickness=2, label=""):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for (x, y, w, h) in bounding_boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
        if label:
            draw.text((x, max(0, y - 15)), label, fill=color)
    return img_with_boxes

def draw_symbol_boxes(image, detections, color_map=None, thickness=2):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    if color_map is None:
        color_map = {"Added": (0, 255, 0), "Removed": (255, 0, 0),
                     "Misplaced": (255, 255, 0), "Symbol": (0, 0, 255)}
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d.get("label", "Symbol")
        color = color_map.get(label, (0, 0, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1, max(0, y1 - 15)), label, fill=color)
    return img_with_boxes

# ============================================================
# FEATURE TABLE DIFF  (original)
# ============================================================

def get_feature_diffs(base_df, comp_df, comp_type, fuzzy_threshold=85):
    if base_df.empty or comp_df.empty:
        return [], []
    base_vals = base_df[base_df['Type'] == comp_type]['Value'].tolist()
    comp_vals = comp_df[comp_df['Type'] == comp_type]['Value'].tolist()
    added, deleted = [], []
    if comp_type in ['Barcode', 'Image']:
        added = list(set(comp_vals) - set(base_vals))
        deleted = list(set(base_vals) - set(comp_vals))
        return added, deleted
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
# OCR CROP  (original — kept for any future per-box use)
# ============================================================

def ocr_crop(image, box):
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
    if len(re.findall(r'[A-Za-z0-9]', text)) < 2:
        return ""
    return text

# ============================================================
# MAIN APP
# ============================================================

st.markdown('<h1 class="main-header">LABEL COMPARATOR PRO</h1>', unsafe_allow_html=True)

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    base_file = st.file_uploader("Upload Base Label",
                                  type=["jpg", "png", "jpeg", "pdf"], key="base")
with col_upload2:
    child_files = st.file_uploader("Upload Child Label(s)",
                                    type=["jpg", "png", "jpeg", "pdf"],
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
            # BASE — processed once, reused for every child
            # ------------------------------------------------------------------
            status.write("Processing base document...")
            raw_base_img     = process_upload(base_file)
            base_processed   = preprocess_image(raw_base_img, enhance_contrast=False)
            base_symbols_raw = run_detection_pil(base_processed)
            base_features_df = extract_all_features(raw_base_img, base_symbols_raw,
                                                     logo_folder="logos")
            # Extract base text lines + positions once
            base_text_lines  = get_text_lines_with_bbox(base_processed)

            tabs = st.tabs([f.name for f in child_files])

            for tab, child_file in zip(tabs, child_files):
                with tab:
                    status.write(f"Analyzing {child_file.name}...")

                    # ----------------------------------------------------------
                    # CHILD
                    # ----------------------------------------------------------
                    raw_child_img    = process_upload(child_file)
                    comp_processed   = preprocess_image(raw_child_img, enhance_contrast=False)
                    comp_aligned, _  = align_images(base_processed, comp_processed)
                    comp_symbols_raw = run_detection_pil(comp_aligned)
                    comp_features_df = extract_all_features(comp_aligned, comp_symbols_raw,
                                                             logo_folder="logos")

                    # Child text lines extracted on original (not aligned) so
                    # different aspect ratios / sizes are handled correctly
                    child_text_lines = get_text_lines_with_bbox(comp_processed)

                    # SSIM score for the similarity metric display
                    diff_results = find_differences(base_processed, comp_aligned)
                    ssim_score   = diff_results['ssim_score'] if diff_results else 0.0

                    # ----------------------------------------------------------
                    # SYMBOL COMPARISON  (original logic)
                    # ----------------------------------------------------------
                    comp_symbols_final = []

                    def region_has_symbol(image, bbox, threshold=15):
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = np.array(image)[y1:y2, x1:x2]
                        if crop.size == 0:
                            return False
                        crop_gray = (cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                                     if len(crop.shape) == 3 else crop)
                        return np.sum(crop_gray < 240) > threshold

                    for base_sym in base_symbols_raw:
                        matches = [c for c in comp_symbols_raw
                                   if c["class"] == base_sym["class"]]
                        if matches:
                            for match in matches:
                                if math.dist(get_center(base_sym["bbox"]),
                                             get_center(match["bbox"])) > 40:
                                    if region_has_symbol(comp_aligned, match["bbox"]):
                                        comp_symbols_final.append(
                                            dict(match, label="Misplaced"))
                        else:
                            comp_symbols_final.append(dict(base_sym, label="Removed"))

                    for d in comp_symbols_raw:
                        if d["class"] not in [b["class"] for b in base_symbols_raw]:
                            comp_symbols_final.append(dict(d, label="Added"))

                    comp_symbols = comp_symbols_final

                    # ----------------------------------------------------------
                    # TEXT DIFF  (feature-position approach)
                    # ----------------------------------------------------------
                    deleted_lines, added_lines, changed_lines = diff_text_lines(
                        base_text_lines, child_text_lines
                    )

                    deleted_boxes       = [l['bbox'] for l in deleted_lines]
                    added_boxes         = [l['bbox'] for l in added_lines]
                    changed_base_boxes  = [pair[0]['bbox'] for pair in changed_lines]
                    changed_child_boxes = [pair[1]['bbox'] for pair in changed_lines]

                    deleted_text  = [l['text'] for l in deleted_lines]
                    added_text    = [l['text'] for l in added_lines]
                    modified_text = [
                        f"From: '{p[0]['text']}' ➔ To: '{p[1]['text']}'"
                        for p in changed_lines
                    ]

                    # ----------------------------------------------------------
                    # DRAW
                    # ----------------------------------------------------------
                    base_marked = draw_differences(base_processed, deleted_boxes,
                                                   color=(255, 0, 0), label="Deleted")
                    base_marked = draw_differences(base_marked, changed_base_boxes,
                                                   color=(255, 165, 0), label="Changed")

                    comp_marked = draw_differences(comp_processed, added_boxes,
                                                   color=(0, 200, 0), label="Added")
                    comp_marked = draw_differences(comp_marked, changed_child_boxes,
                                                   color=(255, 165, 0), label="Changed")
                    comp_marked = draw_symbol_boxes(
                        comp_marked, comp_symbols,
                        color_map={"Added": (0, 200, 0), "Removed": (255, 0, 0),
                                   "Misplaced": (255, 200, 0)}
                    )

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
                        st.dataframe(base_features_df, use_container_width=True,
                                     hide_index=True)
                    with feat_col2:
                        st.markdown("**Child Features**")
                        st.dataframe(comp_features_df, use_container_width=True,
                                     hide_index=True)

                    st.markdown("---")
                    st.markdown("### 📊 Interactive Discrepancy Report")

                    added_bc,  deleted_bc  = get_feature_diffs(base_features_df,
                                                                comp_features_df, 'Barcode')
                    added_img, deleted_img = get_feature_diffs(base_features_df,
                                                                comp_features_df, 'Image')
                    added_syms     = [s["class"] for s in comp_symbols if s["label"] == "Added"]
                    removed_syms   = [s["class"] for s in comp_symbols if s["label"] == "Removed"]
                    misplaced_syms = [s["class"] for s in comp_symbols if s["label"] == "Misplaced"]

                    diff_data = []
                    for item in added_text:     diff_data.append({"Category": "Text",    "Status": "Added",     "Value": item})
                    for item in deleted_text:   diff_data.append({"Category": "Text",    "Status": "Deleted",   "Value": item})
                    for item in modified_text:  diff_data.append({"Category": "Text",    "Status": "Modified",  "Value": item})
                    for item in added_syms:     diff_data.append({"Category": "Symbol",  "Status": "Added",     "Value": item})
                    for item in misplaced_syms: diff_data.append({"Category": "Symbol",  "Status": "Misplaced", "Value": item})
                    for item in removed_syms:   diff_data.append({"Category": "Symbol",  "Status": "Deleted",   "Value": item})
                    for item in added_bc:       diff_data.append({"Category": "Barcode", "Status": "Added",     "Value": item})
                    for item in deleted_bc:     diff_data.append({"Category": "Barcode", "Status": "Deleted",   "Value": item})
                    for item in added_img:      diff_data.append({"Category": "Image",   "Status": "Added",     "Value": item})
                    for item in deleted_img:    diff_data.append({"Category": "Image",   "Status": "Deleted",   "Value": item})

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

                        styled_df = diff_df.style.apply(highlight_status, axis=1)
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No discrepancies found! The labels match perfectly.")

            status.update(label="Analysis Complete!", state="complete", expanded=False)

    else:
        st.error("Please ensure both the Base Label and at least one Child Label are uploaded.")

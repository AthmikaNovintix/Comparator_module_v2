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
import os

from rapidfuzz import fuzz

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

# ---------------------------------------------------------
# UI DATA INJECTION HELPER (Bypasses Extract.py Caching)
# ---------------------------------------------------------
def force_symbols_into_df(features_df, symbols_raw):
    """Guarantees symbols appear in the Extracted Features table"""
    if not symbols_raw:
        return features_df
        
    symbol_entries = []
    seen = set()
    
    if not features_df.empty and 'Type' in features_df.columns:
        seen = set(features_df[features_df['Type'] == 'Symbol']['Value'].tolist())
        
    for sym in symbols_raw:
        val = sym["class"]
        if val not in seen:
            symbol_entries.append({"Type": "Symbol", "Value": val})
            seen.add(val)
            
    if symbol_entries:
        if features_df.empty:
            return pd.DataFrame(symbol_entries)
        else:
            return pd.concat([features_df, pd.DataFrame(symbol_entries)], ignore_index=True)
            
    return features_df

# ---------------------------------------------------------
# TEMPLATE MATCHING SYMBOL DETECTOR (STABLE VERSION)
# ---------------------------------------------------------
def detect_symbols_template(image, symbol_folder="symbols", threshold=0.80):
    if not os.path.exists(symbol_folder):
        return []
        
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        img_gray = img_np
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    
    detections = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for symbol_file in os.listdir(symbol_folder):
        if not symbol_file.lower().endswith(valid_extensions):
            continue
            
        symbol_path = os.path.join(symbol_folder, symbol_file)
        symbol_img = cv2.imread(symbol_path, 0)
        if symbol_img is None: continue
        
        symbol_name = os.path.splitext(symbol_file)[0]
        symbol_enhanced = clahe.apply(symbol_img)
        
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            width = int(symbol_enhanced.shape[1] * scale)
            height = int(symbol_enhanced.shape[0] * scale)
            
            if width < 10 or height < 10 or width > img_enhanced.shape[1] or height > img_enhanced.shape[0]:
                continue
                
            resized_sym = cv2.resize(symbol_enhanced, (width, height), interpolation=cv2.INTER_AREA)
            
            res = cv2.matchTemplate(img_enhanced, resized_sym, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                x1, y1 = pt[0], pt[1]
                x2, y2 = x1 + width, y1 + height
                conf = float(res[y1, x1])
                
                overlap = False
                for d in detections:
                    if d['class'] == symbol_name:
                        dx1, dy1, dx2, dy2 = d['bbox']
                        ixA = max(x1, dx1); iyA = max(y1, dy1)
                        ixB = min(x2, dx2); iyB = min(y2, dy2)
                        interArea = max(0, ixB - ixA) * max(0, iyB - iyA)
                        if interArea > 0:
                            box1Area = (x2 - x1) * (y2 - y1)
                            box2Area = (dx2 - dx1) * (dy2 - dy1)
                            iou = interArea / float(box1Area + box2Area - interArea)
                            if iou > 0.1: 
                                overlap = True
                                if conf > d['confidence']:
                                    d['bbox'] = [x1, y1, x2, y2]
                                    d['confidence'] = conf
                                break
                if not overlap:
                    detections.append({
                        "class": symbol_name,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "label": "Symbol"
                    })
                        
    return detections

def get_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

# ---------------------------------------------------------
# STANDARD PIPELINE
# ---------------------------------------------------------
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
        new_height = int((float(img.height) * float(ratio)))
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    return img

def preprocess_image(image, resize_to=None, enhance_contrast=False):
    if image is None: return None
    img = image.copy()
    if resize_to:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    return img

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
        bounding_boxes = []
        for c in filtered_cnts:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))
        return {
            'ssim_score': score,
            'bounding_boxes': bounding_boxes,
            'total_differences': len(bounding_boxes)
        }
    except Exception as e:
        st.error(f"Error finding differences: {e}")
        return None

def boxes_overlap(boxA, boxB, iou_threshold=0.3):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou > iou_threshold

def filter_text_boxes(text_boxes, symbol_boxes):
    filtered = []
    for (x, y, w, h) in text_boxes:
        text_box = [x, y, x + w, y + h]
        overlap = False
        for sym in symbol_boxes:
            sym_box = sym["bbox"]
            if boxes_overlap(text_box, sym_box):
                overlap = True
                break
        if not overlap:
            filtered.append((x, y, w, h))
    return filtered

def draw_differences(image, bounding_boxes, color=(255, 0, 0), thickness=2, label=""):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for (x, y, w, h) in bounding_boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
        if label:
            draw.text((x, max(0, y-15)), label, fill=color)
    return img_with_boxes

def draw_symbol_boxes(image, detections, color_map=None, thickness=2):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    if color_map is None:
        color_map = {
            "Added": (0,255,0), 
            "Removed": (255,0,0), 
            "Misplaced": (255,165,0),  
            "Present": (0,0,0)  
        }
        
    for d in detections:
        label = d.get("label", "Symbol")
        if label == "Present":
            continue
            
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = color_map.get(label, (0,0,255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1, max(0, y1-15)), label, fill=color)
    return img_with_boxes

def get_feature_diffs(base_df, comp_df, comp_type, fuzzy_threshold=85):
    if base_df.empty or comp_df.empty:
        return [], []
    
    base_vals_list = base_df[base_df['Type'] == comp_type]['Value'].tolist()
    comp_vals_list = comp_df[comp_df['Type'] == comp_type]['Value'].tolist()
    
    added = []
    deleted = []

    if comp_type in ['Barcode', 'Image']:
        base_set = set(base_vals_list)
        comp_set = set(comp_vals_list)
        added = list(comp_set - base_set)
        deleted = list(base_set - comp_set)
        return added, deleted

    for b_val in base_vals_list:
        match_found = False
        norm_b = b_val.lower().strip() 
        for c_val in comp_vals_list:
            if fuzz.token_set_ratio(norm_b, c_val.lower().strip()) >= fuzzy_threshold:
                match_found = True
                break
        if not match_found:
            deleted.append(b_val)

    for c_val in comp_vals_list:
        match_found = False
        norm_c = c_val.lower().strip() 
        for b_val in base_vals_list:
            if fuzz.token_set_ratio(norm_c, b_val.lower().strip()) >= fuzzy_threshold:
                match_found = True
                break
        if not match_found:
            added.append(c_val)

    return added, deleted

def ocr_crop(image, box):
    x, y, w, h = box
    pad = 5
    img_width = image.shape[1] if isinstance(image, np.ndarray) else image.width
    img_height = image.shape[0] if isinstance(image, np.ndarray) else image.height
    
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(img_width, x+w+pad), min(img_height, y+h+pad)
    
    crop = np.array(image)[y1:y2, x1:x2]
    if crop.size == 0: return ""
    
    if len(crop.shape) == 3: gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else: gray = crop
        
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray, lang='eng+fra+deu', config='--psm 6').strip()
    
    text = re.sub(r'[|><_~=«»"*;]', '', text).strip()
    text = re.sub(r'\n+', ' ', text) 
    return text

# --- Main App ---

st.markdown('<h1 class="main-header">LABEL COMPARATOR PRO</h1>', unsafe_allow_html=True)

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    base_file = st.file_uploader("Upload Base Label", type=["jpg", "png", "jpeg", "pdf"], key="base")
with col_upload2:
    child_files = st.file_uploader("Upload Child Label(s)", type=["jpg", "png", "jpeg", "pdf"], key="child", accept_multiple_files=True)

st.write("") 
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    compare_clicked = st.button("Compare labels", use_container_width=True)
st.write("") 

if compare_clicked:
    if base_file is not None and child_files:
        with st.status("Analyzing labels securely...", expanded=True) as status:
            
            status.write("Processing base document...")
            raw_base_img = process_upload(base_file)
            base_processed = preprocess_image(raw_base_img, enhance_contrast=False)
            
            base_symbols_raw = detect_symbols_template(base_processed, symbol_folder="symbols")
            base_features_df = extract_all_features(raw_base_img, base_symbols_raw, logo_folder="logos")
            
            # CRITICAL FIX: Force inject symbols into DataFrame to bypass Streamlit Caching bugs
            base_features_df = force_symbols_into_df(base_features_df, base_symbols_raw)
            
            base_symbols = []
            for d in base_symbols_raw:
                d = d.copy()
                d["label"] = "Symbol"
                base_symbols.append(d)
                
            tabs = st.tabs([f.name for f in child_files])
            
            for tab, child_file in zip(tabs, child_files):
                with tab:
                    status.write(f"Analyzing {child_file.name}...")
                    raw_child_img = process_upload(child_file)
                    comp_processed = preprocess_image(raw_child_img, enhance_contrast=False)
                    
                    comp_aligned, aligned_success = align_images(base_processed, comp_processed)
                    if not aligned_success:
                        comp_aligned = comp_processed
                    
                    comp_symbols_raw = detect_symbols_template(comp_aligned, symbol_folder="symbols")
                    comp_features_df = extract_all_features(comp_aligned, comp_symbols_raw, logo_folder="logos")
                    
                    # CRITICAL FIX: Force inject symbols into DataFrame to bypass Streamlit Caching bugs
                    comp_features_df = force_symbols_into_df(comp_features_df, comp_symbols_raw)
                        
                    diff_results = find_differences(base_processed, comp_aligned, threshold=0.85, min_area=150)
                    if not diff_results:
                        st.error(f"Error comparing '{child_file.name}'")
                        continue
                        
                    comp_symbols_final = []
                    dynamic_threshold = base_processed.width * 0.05 
                    claimed_child_indices = set()
                    
                    for base_sym in base_symbols_raw:
                        best_match = None
                        min_dist = float('inf')
                        best_idx = -1
                        
                        for idx, c_sym in enumerate(comp_symbols_raw):
                            if c_sym["class"] == base_sym["class"] and idx not in claimed_child_indices:
                                c1 = get_center(base_sym["bbox"])
                                c2 = get_center(c_sym["bbox"])
                                dist = math.dist(c1, c2)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = c_sym
                                    best_idx = idx
                                    
                        if best_match is not None:
                            claimed_child_indices.add(best_idx)
                            if min_dist > dynamic_threshold:
                                misplaced_box = best_match.copy()
                                misplaced_box["label"] = "Misplaced"
                                comp_symbols_final.append(misplaced_box)
                            else:
                                present_box = best_match.copy()
                                present_box["label"] = "Present"
                                comp_symbols_final.append(present_box)
                        else:
                            missing_box = base_sym.copy()
                            missing_box["label"] = "Removed"
                            comp_symbols_final.append(missing_box)

                    for idx, c_sym in enumerate(comp_symbols_raw):
                        if idx not in claimed_child_indices:
                            added_box = c_sym.copy()
                            added_box["label"] = "Added"
                            comp_symbols_final.append(added_box)
                            
                    comp_symbols = comp_symbols_final
                    
                    ssim_boxes = diff_results['bounding_boxes']
                    text_diff_boxes = []
                    for box in ssim_boxes:
                        overlap = False
                        box_coords = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                        for sym in base_symbols:
                            if boxes_overlap(box_coords, sym["bbox"]): overlap = True; break
                        for sym in comp_symbols:
                            if boxes_overlap(box_coords, sym["bbox"]): overlap = True; break
                        if not overlap:
                            text_diff_boxes.append(box)

                    actual_deleted_boxes = []
                    actual_added_boxes = []
                    changed_boxes = []

                    base_gray = cv2.cvtColor(np.array(base_processed), cv2.COLOR_RGB2GRAY)
                    child_gray = cv2.cvtColor(np.array(comp_aligned), cv2.COLOR_RGB2GRAY)

                    for (x, y, w, h) in text_diff_boxes:
                        crop_b = base_gray[y:y+h, x:x+w]
                        crop_c = child_gray[y:y+h, x:x+w]
                        
                        if crop_b.size == 0 or crop_c.size == 0: continue
                        
                        dark_pixels_b = np.sum(crop_b < 220)
                        dark_pixels_c = np.sum(crop_c < 220)
                        
                        min_pixels = 15 
                        has_content_b = dark_pixels_b > min_pixels
                        has_content_c = dark_pixels_c > min_pixels
                        
                        if has_content_b and not has_content_c:
                            actual_deleted_boxes.append((x, y, w, h))
                        elif not has_content_b and has_content_c:
                            actual_added_boxes.append((x, y, w, h))
                        elif has_content_b and has_content_c:
                            changed_boxes.append((x, y, w, h))

                    removed_symbols = [s for s in comp_symbols if s["label"] == "Removed"]
                    added_misplaced_symbols = [s for s in comp_symbols if s["label"] in ["Added", "Misplaced"]]

                    base_marked = draw_differences(base_processed, actual_deleted_boxes, color=(255,0,0), label="Deleted")
                    base_marked = draw_differences(base_marked, changed_boxes, color=(23,162,184), label="Modified")
                    base_marked = draw_symbol_boxes(base_marked, removed_symbols, color_map={"Removed": (255,0,0)})
                    
                    comp_marked = draw_differences(comp_aligned, actual_added_boxes, color=(0,255,0), label="Added")
                    comp_marked = draw_differences(comp_marked, changed_boxes, color=(23,162,184), label="Modified")
                    comp_marked = draw_symbol_boxes(comp_marked, added_misplaced_symbols, color_map={"Added": (0,255,0), "Misplaced": (255,165,0)})

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

                    added_text = []
                    for box in actual_added_boxes:
                        txt = ocr_crop(comp_aligned, box)
                        if txt and len(txt) > 2: added_text.append(txt)

                    deleted_text = []
                    for box in actual_deleted_boxes:
                        txt = ocr_crop(base_processed, box)
                        if txt and len(txt) > 2: deleted_text.append(txt)

                    modified_text = []
                    for box in changed_boxes:
                        txt_b = ocr_crop(base_processed, box)
                        txt_c = ocr_crop(comp_aligned, box)
                        if txt_b or txt_c:
                            modified_text.append(f"From: '{txt_b}' ➔ To: '{txt_c}'")

                    added_bc, deleted_bc = get_feature_diffs(base_features_df, comp_features_df, 'Barcode')
                    added_img, deleted_img = get_feature_diffs(base_features_df, comp_features_df, 'Image')

                    added_syms = [s["class"] for s in comp_symbols if s["label"] == "Added"]
                    removed_syms = [s["class"] for s in comp_symbols if s["label"] == "Removed"]
                    misplaced_syms = [s["class"] for s in comp_symbols if s["label"] == "Misplaced"]
                    
                    st.markdown("---")
                    st.markdown("### 📊 Interactive Discrepancy Report")
                    
                    diff_data = []
                    
                    for item in added_text: diff_data.append({"Category": "Text", "Status": "Added", "Value": item})
                    for item in deleted_text: diff_data.append({"Category": "Text", "Status": "Deleted", "Value": item})
                    for item in modified_text: diff_data.append({"Category": "Text", "Status": "Modified", "Value": item})
                    
                    for item in added_syms: diff_data.append({"Category": "Symbol", "Status": "Added", "Value": item})
                    for item in misplaced_syms: diff_data.append({"Category": "Symbol", "Status": "Misplaced", "Value": item})
                    for item in removed_syms: diff_data.append({"Category": "Symbol", "Status": "Deleted", "Value": item})
                    
                    for item in added_bc: diff_data.append({"Category": "Barcode", "Status": "Added", "Value": item})
                    for item in deleted_bc: diff_data.append({"Category": "Barcode", "Status": "Deleted", "Value": item})
                    
                    for item in added_img: diff_data.append({"Category": "Image", "Status": "Added", "Value": item})
                    for item in deleted_img: diff_data.append({"Category": "Image", "Status": "Deleted", "Value": item})

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

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

# --- NEW DEPENDENCY FOR FUZZY MATCHING ---
from rapidfuzz import fuzz

# External imports from your ML modules (ensure extractor.py is present)
from detect import run_detection_pil, get_center
try:
    from Extract import extract_all_features
except ImportError as e:
    # Fallback if extractor isn't ready, but this fix requires it.
    st.error(f"Error loading {e}")
    st.stop()

# Page configuration
st.set_page_config(layout="wide", page_title="Label Comparator")

# CSS and general layout functions
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp { background-color: #f8f9fa; font-family: 'Roboto', sans-serif; }
        h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm, [data-testid="stMarkdownContainer"] p strong, .stMarkdown p { color: #064b75 !important; }
        .main-header { text-align: center; color: #064b75 !important; font-weight: 700; padding-bottom: 30px; }
        div.stButton > button { background-color: #f4a303 !important; color: #ffffff !important; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; padding: 10px 24px; }
        div.stButton > button:hover { background-color: #e09600 !important; color: white !important; }
        [data-testid="stFileUploader"] { background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px; }
        .results-text { color: #333333; font-size: 16px; line-height: 1.6; }
        .results-text-title { color: #064b75; font-weight: bold; margin-top: 15px; margin-bottom: 5px; }
        .highlight-green { color: green; font-weight: bold; }
        .highlight-red { color: red; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

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

def process_upload(uploaded_file, max_width=1500):
    if uploaded_file is None:
        return None
    if uploaded_file.name.lower().endswith(".pdf"):
        img = pdf_to_image(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
    # Resize if the image is massive to save processing time
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int((float(img.height) * float(ratio)))
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    return img

# --- Image Processing ---
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
    """
    Finds differences, slightly relaxing sensitivity to reduce phantom boxes.
    Updated default threshold (0.8 -> 0.85) and min_area (100 -> 150).
    """
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

# --- Visualization ---
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
            "Added": (0,255,0), # Green
            "Removed": (255,0,0), # Red
            "Misplaced": (255,255,0), # Yellow
            "Symbol": (0,0,255) # Blue
        }
        
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d.get("label", "Symbol")
        color = color_map.get(label, (0,0,255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1, max(0, y1-15)), label, fill=color)
    return img_with_boxes

# --- Helper for Diffing Features with NEW FUZZY LOGIC ---
def get_feature_diffs(base_df, comp_df, comp_type, fuzzy_threshold=95):
    """
    Master function to compare component values. Utilizes Fuzzy Matching 
    for 'Text' type to handle OCR noise/jitter, solving 'Added AND Deleted' issues.
    """
    if base_df.empty or comp_df.empty:
        return [], []
    
    # Get raw lists for Text, sets for exact matching (BC/Image)
    base_vals_list = base_df[base_df['Type'] == comp_type]['Value'].tolist()
    comp_vals_list = comp_df[comp_df['Type'] == comp_type]['Value'].tolist()
    
    added = []
    deleted = []

    # 1. Handle Barcodes and Images exactly (No OCR noise here)
    if comp_type in ['Barcode', 'Image']:
        base_set = set(base_vals_list)
        comp_set = set(comp_vals_list)
        added = list(comp_set - base_set)
        deleted = list(base_set - comp_set)
        return added, deleted

    # 2. Handle Text with Deep Level Fuzzy Matching
    # Find Deleted (Items in Base NOT sufficiently matched in Child)
    for b_val in base_vals_list:
        match_found = False
        norm_b = b_val.lower().strip() # Base normalization
        for c_val in comp_vals_list:
            # Check for high similarity score (95%+ match)
            if fuzz.ratio(norm_b, c_val.lower().strip()) >= fuzzy_threshold:
                match_found = True
                break
        if not match_found:
            deleted.append(b_val)

    # Find Added (Items in Child NOT sufficiently matched in Base)
    for c_val in comp_vals_list:
        match_found = False
        norm_c = c_val.lower().strip() # Child normalization
        for b_val in base_vals_list:
            # Check for high similarity score
            if fuzz.ratio(norm_c, b_val.lower().strip()) >= fuzzy_threshold:
                match_found = True
                break
        if not match_found:
            added.append(c_val)

    return added, deleted

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
        # Use st.status for better progress feedback
        with st.status("Analyzing labels at deep level...", expanded=True) as status:
            
            # 1. Process Base
            status.write("Processing base document and extracting features...")
            raw_base_img = process_upload(base_file)
            base_processed = preprocess_image(raw_base_img, enhance_contrast=False)
            base_symbols_raw = run_detection_pil(base_processed)
            
            # Run the new master extraction (extractor.py)
            base_features_df = extract_all_features(raw_base_img, logo_folder="logos")
            
            base_symbols = []
            for d in base_symbols_raw:
                d = d.copy()
                d["label"] = "Symbol"
                base_symbols.append(d)
                
            # Process each Child
            tabs = st.tabs([f.name for f in child_files])
            
            for tab, child_file in zip(tabs, child_files):
                with tab:
                    status.write(f"Analyzing {child_file.name}...")
                    raw_child_img = process_upload(child_file)
                    comp_processed = preprocess_image(raw_child_img, enhance_contrast=False)
                    
                    # Run the new master extraction on child
                    comp_features_df = extract_all_features(raw_child_img, logo_folder="logos")
                    
                    # Align child to base
                    comp_aligned, aligned_success = align_images(base_processed, comp_processed)
                    if not aligned_success:
                        comp_aligned = comp_processed
                        
                    # SSIM Differences (using slightly relaxed visual threshold)
                    diff_results = find_differences(base_processed, comp_aligned, threshold=0.85, min_area=150)
                    
                    if not diff_results:
                        st.error(f"Error comparing '{child_file.name}'")
                        continue
                        
                    # ML Symbol Comparison
                    comp_symbols_raw = run_detection_pil(comp_aligned)
                    comp_symbols_final = []
                    
                    def region_has_symbol(image, bbox, threshold=15):
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = np.array(image)[y1:y2, x1:x2]
                        if crop.size == 0: return False
                        if len(crop.shape) == 3:
                            crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                        else:
                            crop_gray = crop
                        non_bg = np.sum(crop_gray < 240)
                        return non_bg > threshold

                    for base_sym in base_symbols_raw:
                        matches = [c for c in comp_symbols_raw if c["class"] == base_sym["class"]]
                        if matches:
                            for match in matches:
                                c1 = get_center(base_sym["bbox"])
                                c2 = get_center(match["bbox"])
                                dist = math.dist(c1, c2)
                                if dist > 40: 
                                    if region_has_symbol(comp_aligned, match["bbox"]):
                                        misplaced_box = match.copy()
                                        misplaced_box["label"] = "Misplaced"
                                        comp_symbols_final.append(misplaced_box)
                        else:
                            missing_box = base_sym.copy()
                            missing_box["label"] = "Removed"
                            comp_symbols_final.append(missing_box)
                            
                    for d in comp_symbols_raw:
                        if d["class"] not in [b["class"] for b in base_symbols_raw]:
                            added_box = d.copy()
                            added_box["label"] = "Added"
                            comp_symbols_final.append(added_box)
                            
                    comp_symbols = comp_symbols_final
                    
                    # --- NEW INTELLIGENT DIFFERENCE CATEGORIZATION ---
                    ssim_boxes = diff_results['bounding_boxes']
                    
                    # 1. Filter out areas that YOLO already identified as symbols
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

                    # 2. Categorize the remaining text boxes by checking pixel density
                    actual_deleted_boxes = []
                    actual_added_boxes = []
                    changed_boxes = []

                    base_gray = cv2.cvtColor(np.array(base_processed), cv2.COLOR_RGB2GRAY)
                    child_gray = cv2.cvtColor(np.array(comp_aligned), cv2.COLOR_RGB2GRAY)

                    for (x, y, w, h) in text_diff_boxes:
                        # Crop region from both images
                        crop_b = base_gray[y:y+h, x:x+w]
                        crop_c = child_gray[y:y+h, x:x+w]
                        
                        if crop_b.size == 0 or crop_c.size == 0: continue
                        
                        # Count dark pixels (assuming dark text on light background)
                        # Threshold 220: Anything darker than light gray is considered "content"
                        dark_pixels_b = np.sum(crop_b < 220)
                        dark_pixels_c = np.sum(crop_c < 220)
                        
                        min_pixels = 15 # Ignore tiny specks of dust/noise
                        has_content_b = dark_pixels_b > min_pixels
                        has_content_c = dark_pixels_c > min_pixels
                        
                        if has_content_b and not has_content_c:
                            actual_deleted_boxes.append((x, y, w, h))
                        elif not has_content_b and has_content_c:
                            actual_added_boxes.append((x, y, w, h))
                        elif has_content_b and has_content_c:
                            # Content exists in both, meaning the text changed/shifted
                            changed_boxes.append((x, y, w, h))

                    # 3. Draw Discrepancies Intelligently
                    # Base Image gets Deleted (Red) and Changed (Orange)
                    base_marked = draw_differences(base_processed, actual_deleted_boxes, color=(255,0,0), label="Deleted")
                    base_marked = draw_differences(base_marked, changed_boxes, color=(255,165,0), label="Changed")
                    
                    # Child Image gets Added (Green) and Changed (Orange)
                    comp_marked = draw_differences(comp_aligned, actual_added_boxes, color=(0,255,0), label="Added")
                    comp_marked = draw_differences(comp_marked, changed_boxes, color=(255,165,0), label="Changed")
                    
                    # Finally, draw the ML Symbol boxes over the top
                    comp_marked = draw_symbol_boxes(comp_marked, comp_symbols, color_map={"Added": (0,255,0), "Removed": (255,0,0), "Misplaced": (255,255,0)})

                    # --- Display UI ---
                    st.markdown("### Visual Comparison")
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown("**Label B (Base)**")
                        st.image(base_marked, use_container_width=True)
                        
                    with img_col2:
                        st.markdown(f"**Label C (Child: {child_file.name})**")
                        st.image(comp_marked, use_container_width=True)
                        
                    # Extracted Features
                    st.markdown("---")
                    st.markdown("### Feature Extracted Tables")
                    
                    feat_col1, feat_col2 = st.columns(2)
                    with feat_col1:
                        st.markdown("**Base Features**")
                        st.dataframe(base_features_df, use_container_width=True, hide_index=True)
                    with feat_col2:
                        st.markdown("**Child Features**")
                        st.dataframe(comp_features_df, use_container_width=True, hide_index=True)
                        
                    # Calculate Differences for Output utilizing the NEW get_feature_diffs
                    # Parameter sets fuzzy matching threshold (95%+ match)

                    added_text, deleted_text = get_feature_diffs(base_features_df, comp_features_df, 'Text', fuzzy_threshold=95)
                    added_bc, deleted_bc = get_feature_diffs(base_features_df, comp_features_df, 'Barcode')
                    added_img, deleted_img = get_feature_diffs(base_features_df, comp_features_df, 'Image')

                    added_syms = [s["class"] for s in comp_symbols if s["label"] == "Added"]
                    removed_syms = [s["class"] for s in comp_symbols if s["label"] == "Removed"]
                    misplaced_syms = [s["class"] for s in comp_symbols if s["label"] == "Misplaced"]
                    
                    # Comparison Text
                    st.markdown("---")
                    st.markdown("### Discrepancy Report (Non-Tabular)")
                    
                    # --- NEW VISUAL DISCREPANCY REPORT ---
                    st.markdown("---")
                    st.markdown("### 📊 Interactive Discrepancy Report")
                    
                    # 1. Compile all differences into a structured list
                    diff_data = []
                    
                    for item in added_text: diff_data.append({"Category": "Text", "Status": "Added", "Value": item})
                    for item in deleted_text: diff_data.append({"Category": "Text", "Status": "Deleted", "Value": item})
                    
                    for item in added_syms: diff_data.append({"Category": "Symbol", "Status": "Added", "Value": item})
                    for item in misplaced_syms: diff_data.append({"Category": "Symbol", "Status": "Misplaced", "Value": item})
                    for item in removed_syms: diff_data.append({"Category": "Symbol", "Status": "Deleted", "Value": item})
                    
                    for item in added_bc: diff_data.append({"Category": "Barcode", "Status": "Added", "Value": item})
                    for item in deleted_bc: diff_data.append({"Category": "Barcode", "Status": "Deleted", "Value": item})
                    
                    for item in added_img: diff_data.append({"Category": "Image", "Status": "Added", "Value": item})
                    for item in deleted_img: diff_data.append({"Category": "Image", "Status": "Deleted", "Value": item})

                    # 2. Convert to DataFrame and apply beautiful color styling
                    if diff_data:
                        diff_df = pd.DataFrame(diff_data)
                        
                        def highlight_status(row):
                            if row['Status'] == 'Added':
                                return ['background-color: rgba(40, 167, 69, 0.2); color: #155724'] * len(row)
                            elif row['Status'] == 'Deleted':
                                return ['background-color: rgba(220, 53, 69, 0.2); color: #721c24'] * len(row)
                            elif row['Status'] == 'Misplaced':
                                return ['background-color: rgba(255, 193, 7, 0.2); color: #856404'] * len(row)
                            return [''] * len(row)
                        
                        styled_df = diff_df.style.apply(highlight_status, axis=1)
                        
                        # Display as an interactive dataframe
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No discrepancies found! The labels match perfectly.")
            status.update(label="Deep Analysis Complete!", state="complete", expanded=False)

    else:
        st.error("Please ensure both the Base Label and at least one Child Label are uploaded.")

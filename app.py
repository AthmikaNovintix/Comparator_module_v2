import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re

from rapidfuzz import fuzz
from detect import run_detection_pil

try:
    from Extract import extract_all_features
except ImportError as e:
    st.error(f"Error loading external module: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="Label Comparator Pro")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp { background-color: #f8f9fa; font-family: 'Roboto', sans-serif; }
        h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm, [data-testid="stMarkdownContainer"] p strong, .stMarkdown p { color: #064b75 !important; }
        .main-header { text-align: center; color: #064b75 !important; font-weight: 700; padding-bottom: 30px; }
        div.stButton > button { background-color: #f4a303 !important; color: #ffffff !important; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; padding: 10px 24px; }
        div.stButton > button:hover { background-color: #e09600 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. IMAGE PROCESSING
# -------------------------------------------------------------------
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
    if image.mode != 'RGB': image = image.convert('RGB')
    pdf_document.close()
    return image

def process_upload(uploaded_file, target_width=1200):
    if uploaded_file is None: return None
    if uploaded_file.name.lower().endswith(".pdf"):
        img = pdf_to_image(uploaded_file)
    else:
        img = Image.open(uploaded_file)
        if img.mode != 'RGB': img = img.convert('RGB')
            
    ratio = target_width / float(img.width)
    new_height = int((float(img.height) * float(ratio)))
    return img.resize((target_width, new_height), Image.Resampling.LANCZOS)

# -------------------------------------------------------------------
# 2. SEMANTIC TEXT EXTRACTION (Layout Agnostic)
# -------------------------------------------------------------------
@st.cache_data
def get_text_blocks(_image):
    """Extracts text and absolute bounding boxes independent of image size"""
    gray = cv2.cvtColor(np.array(_image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    data = pytesseract.image_to_data(gray, output_type=Output.DICT, lang='eng+fra+deu', config='--psm 3')
    
    lines = {}
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        # Only accept highly confident text to eliminate noise
        if conf > 40 and len(text) > 1:
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            key = f"{block_num}_{line_num}"
            
            # Scale coordinates back down to original image size
            x1 = int(data['left'][i] * 0.5)
            y1 = int(data['top'][i] * 0.5)
            x2 = int((data['left'][i] + data['width'][i]) * 0.5)
            y2 = int((data['top'][i] + data['height'][i]) * 0.5)
            
            if key not in lines:
                lines[key] = {"text": text, "bbox": [x1, y1, x2, y2]}
            else:
                lines[key]["text"] += " " + text
                # Expand bounding box to fit the entire line
                lines[key]["bbox"][0] = min(lines[key]["bbox"][0], x1)
                lines[key]["bbox"][1] = min(lines[key]["bbox"][1], y1)
                lines[key]["bbox"][2] = max(lines[key]["bbox"][2], x2)
                lines[key]["bbox"][3] = max(lines[key]["bbox"][3], y2)
                
    result = []
    for v in lines.values():
        clean_text = re.sub(r'[|><_~=«»"*;]', '', v["text"]).strip()
        if len(clean_text) > 2:
            result.append({"text": clean_text, "bbox": v["bbox"]})
            
    return result

# -------------------------------------------------------------------
# 3. SEMANTIC MATCHING ENGINES
# -------------------------------------------------------------------
def match_semantic_text(base_blocks, child_blocks):
    added, deleted, modified = [], [], []
    matched_child_indices = set()
    
    for b_block in base_blocks:
        best_match = None
        best_ratio = 0
        best_idx = -1
        
        for idx, c_block in enumerate(child_blocks):
            if idx in matched_child_indices: continue
            
            ratio = fuzz.token_set_ratio(b_block["text"].lower(), c_block["text"].lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = c_block
                best_idx = idx
                
        if best_ratio >= 90:
            matched_child_indices.add(best_idx) # Perfect match
        elif 70 <= best_ratio < 90:
            matched_child_indices.add(best_idx) # Modified
            modified.append({"base": b_block, "child": best_match})
        else:
            deleted.append(b_block) # Deleted
            
    for idx, c_block in enumerate(child_blocks):
        if idx not in matched_child_indices:
            added.append(c_block)
            
    return added, deleted, modified

def match_semantic_symbols(base_syms, child_syms):
    """Matches symbols strictly by class count, ignoring coordinates completely"""
    added, deleted = [], []
    matched_child_indices = set()
    
    for b_sym in base_syms:
        matched = False
        for idx, c_sym in enumerate(child_syms):
            if idx in matched_child_indices: continue
            if b_sym["class"] == c_sym["class"]:
                matched_child_indices.add(idx)
                matched = True
                break
        if not matched:
            deleted.append(b_sym)
            
    for idx, c_sym in enumerate(child_syms):
        if idx not in matched_child_indices:
            added.append(c_sym)
            
    return added, deleted

def get_feature_diffs(base_df, comp_df, comp_type):
    if base_df.empty or comp_df.empty: return [], []
    base_set = set(base_df[base_df['Type'] == comp_type]['Value'].tolist())
    comp_set = set(comp_df[comp_df['Type'] == comp_type]['Value'].tolist())
    return list(comp_set - base_set), list(base_set - comp_set)

# -------------------------------------------------------------------
# 4. VISUALIZATION
# -------------------------------------------------------------------
def draw_semantic_boxes(image, boxes, color, label_key="text", label_prefix=""):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        display_text = f"{label_prefix} {item.get(label_key, '')}"[:25]
        draw.text((x1, max(0, y1-15)), display_text, fill=color)
    return img_copy

# --- Main App Execution ---

st.markdown('<h1 class="main-header">LAYOUT AGNOSTIC COMPARATOR</h1>', unsafe_allow_html=True)

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    base_file = st.file_uploader("Upload Base Label", type=["jpg", "png", "jpeg", "pdf"], key="base")
with col_upload2:
    child_files = st.file_uploader("Upload Child Label(s)", type=["jpg", "png", "jpeg", "pdf"], key="child", accept_multiple_files=True)

st.write("") 
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    compare_clicked = st.button("Compare labels", use_container_width=True)

if compare_clicked:
    if base_file is not None and child_files:
        with st.status("Performing Semantic Layout-Agnostic Analysis...", expanded=True) as status:
            
            # --- BASE PROCESSING ---
            status.write("Mapping Base Layout...")
            base_img = process_upload(base_file)
            base_text_blocks = get_text_blocks(base_img)
            base_symbols = run_detection_pil(base_img)
            base_features_df = extract_all_features(base_img, base_symbols, logo_folder="logos")
            
            tabs = st.tabs([f.name for f in child_files])
            
            for tab, child_file in zip(tabs, child_files):
                with tab:
                    # --- CHILD PROCESSING ---
                    status.write(f"Mapping Child Layout for {child_file.name}...")
                    child_img = process_upload(child_file)
                    child_text_blocks = get_text_blocks(child_img)
                    child_symbols = run_detection_pil(child_img)
                    child_features_df = extract_all_features(child_img, child_symbols, logo_folder="logos")
                    
                    # --- SEMANTIC COMPARISON (NO PIXEL MATH) ---
                    status.write("Correlating Semantic Maps...")
                    text_add, text_del, text_mod = match_semantic_text(base_text_blocks, child_text_blocks)
                    sym_add, sym_del = match_semantic_symbols(base_symbols, child_symbols)
                    bc_add, bc_del = get_feature_diffs(base_features_df, child_features_df, 'Barcode')
                    img_add, img_del = get_feature_diffs(base_features_df, child_features_df, 'Image')

                    # --- VISUAL RENDERING ---
                    base_render = base_img.copy()
                    child_render = child_img.copy()

                    # Draw Text Diffs
                    base_render = draw_semantic_boxes(base_render, text_del, (255, 0, 0), "text", "DEL:")
                    child_render = draw_semantic_boxes(child_render, text_add, (0, 255, 0), "text", "NEW:")
                    
                    for mod in text_mod:
                        base_render = draw_semantic_boxes(base_render, [mod["base"]], (255, 165, 0), "text", "MOD:")
                        child_render = draw_semantic_boxes(child_render, [mod["child"]], (255, 165, 0), "text", "MOD:")

                    # Draw Symbol Diffs
                    base_render = draw_semantic_boxes(base_render, sym_del, (255, 0, 0), "class", "DEL:")
                    child_render = draw_semantic_boxes(child_render, sym_add, (0, 255, 0), "class", "NEW:")

                    st.markdown("---")
                    st.markdown("### Visual Comparison (Layout Agnostic)")
                    st.info("Boxes are generated based on data content, not pixel placement. The app is immune to layout shifts.")
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        st.markdown("**Label B (Base)**")
                        st.image(base_render, use_container_width=True)
                    with img_col2:
                        st.markdown(f"**Label C (Child: {child_file.name})**")
                        st.image(child_render, use_container_width=True)
                        
                    # --- REPORT GENERATION ---
                    st.markdown("---")
                    st.markdown("### 📊 Absolute Discrepancy Report")
                    
                    diff_data = []
                    
                    for t in text_add: diff_data.append({"Category": "Text", "Status": "Added", "Value": t["text"]})
                    for t in text_del: diff_data.append({"Category": "Text", "Status": "Deleted", "Value": t["text"]})
                    for t in text_mod: diff_data.append({"Category": "Text", "Status": "Modified", "Value": f"From: {t['base']['text']} ➔ To: {t['child']['text']}"})
                    
                    for s in sym_add: diff_data.append({"Category": "Symbol", "Status": "Added", "Value": s["class"]})
                    for s in sym_del: diff_data.append({"Category": "Symbol", "Status": "Deleted", "Value": s["class"]})
                    
                    for b in bc_add: diff_data.append({"Category": "Barcode", "Status": "Added", "Value": b})
                    for b in bc_del: diff_data.append({"Category": "Barcode", "Status": "Deleted", "Value": b})
                    
                    for i in img_add: diff_data.append({"Category": "Image", "Status": "Added", "Value": i})
                    for i in img_del: diff_data.append({"Category": "Image", "Status": "Deleted", "Value": i})

                    if diff_data:
                        diff_df = pd.DataFrame(diff_data)
                        def highlight_status(row):
                            if row['Status'] == 'Added': return ['background-color: rgba(40, 167, 69, 0.2); color: #155724'] * len(row)
                            if row['Status'] == 'Deleted': return ['background-color: rgba(220, 53, 69, 0.2); color: #721c24'] * len(row)
                            if row['Status'] == 'Modified': return ['background-color: rgba(255, 193, 7, 0.2); color: #856404'] * len(row)
                            return [''] * len(row)
                        
                        styled_df = diff_df.style.apply(highlight_status, axis=1)
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No semantic discrepancies found! All required data is present.")
                        
            status.update(label="Semantic Analysis Complete!", state="complete", expanded=False)

    else:
        st.error("Please ensure both the Base Label and at least one Child Label are uploaded.")

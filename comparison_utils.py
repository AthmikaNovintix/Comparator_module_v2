import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import cv2
import fitz  # PyMuPDF for PDF handling
import io
import math
from skimage.metrics import structural_similarity as ssim
import imutils

# --- IMPORT MODULAR ML LOGIC ---
from detect import run_detection_pil, compare_labels, get_center

st.set_page_config(layout="wide", page_title="Document Comparison Tool")

# --- Custom CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp {color:#fff;background:#064b75;font-family:'Roboto',sans-serif;}
        [data-testid=stHeader] {color:#f4a303;background:#064b75;}
        [data-testid=stBaseButton-secondary] {background:#f4a303;color:#023C59;}
        [data-testid=stFileUploaderDropzone] {
            background:#f7f7f7;border:2px dashed #4f7cae;border-radius:10px;
            padding:20px;color:#2d4059;
            font-weight:bold;font-size:18px;height:200px;
        }
        [data-testid=stWidgetLabel] {color:#f4a303;}
        .title-container {
            display:flex;justify-content:center;text-align:center;margin:auto;width:655px;
        }
        .comparison-container {
            background: rgba(244, 163, 3, 0.1);
            border: 2px solid #f4a303;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
        }
        .metric-container {
            background: rgba(45, 64, 89, 0.8);
            border: 1px solid #4f7cae;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] span{
            color:#000000;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
header_html = """
    <div class="title-container">
        <span style="font-size:31px;font-weight:bold;color:#fff;">Document</span>
        <span style="font-size:31px;font-weight:bold;color:#f4a303;margin-left:10px;">Comparison Tool</span>
    </div>
    <div style="text-align:center;margin-top:10px;color:#ccc;">
        <p>Compare images and PDFs pixel-by-pixel with difference highlighting</p>
    </div>
"""
with st.container(border=True):
    st.markdown(header_html, unsafe_allow_html=True)

# --- PDF to Image Conversion ---
@st.cache_data
def pdf_to_image(pdf_file, page_num=0, dpi=200):
    """Convert PDF page to PIL Image"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = pdf_document[0]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = first_page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        total_pages = len(pdf_document)
        pdf_document.close()
        return image, total_pages
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None, 0

# --- Image Preprocessing ---
def preprocess_image(image, resize_to=None, enhance_contrast=False):
    """Preprocess image for better comparison"""
    if image is None:
        return None
    img = image.copy()
    if resize_to:
        img = img.resize(resize_to, Image.Resampling.LANCZOS)
    if enhance_contrast:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    return img

# --- Image Alignment ---
def align_images(imageA, imageB, max_features=500, good_match_percent=0.15):
    """Align two images using ORB feature matching"""
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

# --- Difference Detection ---
def find_differences(imageA, imageB, threshold=0.8, min_area=100):
    """Find differences between two images using SSIM"""
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
            'difference_image': diff,
            'threshold_image': thresh,
            'bounding_boxes': bounding_boxes,
            'total_differences': len(bounding_boxes)
        }
    except Exception as e:
        st.error(f"Error finding differences: {e}")
        return None

# --- Visualization ---
def draw_differences(image, bounding_boxes, color=(255, 0, 0), thickness=3):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)
        text = f"Diff {i+1}"
        draw.text((x, max(0, y-20)), text, fill=color)
    return img_with_boxes

def draw_symbol_boxes(image, detections, color_map=None, thickness=3):
    """Draw bounding boxes for symbols detected by ML model"""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    if color_map is None:
        color_map = {
            "Added": (0,255,0),
            "Removed": (255,0,0),
            "Misplaced": (255,255,0),
            "Symbol": (0,0,255)
        }
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d.get("label", "Symbol")
        color = color_map.get(label, (0,0,255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        draw.text((x1, max(0, y1-20)), label, fill=color)
    return img_with_boxes

def create_side_by_side_comparison(imageA, imageB, bounding_boxes, title="Comparison"):
    """Create side-by-side comparison with differences marked"""
    imgA_marked = draw_differences(imageA, bounding_boxes, color=(255, 0, 0))
    imgB_marked = draw_differences(imageB, bounding_boxes, color=(255, 0, 0))
    return imgA_marked, imgB_marked

def image_to_bytes(image):
    """Convert PIL Image to bytes"""
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()

# --- Utility Functions for Layout/Box Processing ---
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

# --- Main Application ---
def main():
    ss = st.session_state

    if 'base_image' not in ss:
        ss.base_image = None
    if 'comparison_images' not in ss:
        ss.comparison_images = []
    if 'results' not in ss:
        ss.results = []

    with st.sidebar:
        st.header("⚙️ Comparison Settings")

        st.subheader("Detection Parameters")
        threshold = st.slider("SSIM Threshold", 0.0, 1.0, 0.8, 0.01, 
                             help="Higher values = more sensitive to differences")
        min_area = st.slider("Minimum Difference Area", 10, 1000, 100, 10,
                           help="Minimum pixel area for difference detection")

        st.subheader("Image Processing")
        enhance_contrast = st.checkbox("Enhance Contrast", help="Improve difference detection")
        align_images_option = st.checkbox("Auto-align Images", value=True, 
                                        help="Automatically align images before comparison")

        st.subheader("Export Options")
        include_difference_images = st.checkbox("Include Difference Images", value=True)
        export_format = st.selectbox("Export Format", ["PNG", "JPEG", "PDF"])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📎 Base Document")
        base_file = st.file_uploader(
            "Upload base image or PDF", 
            type=["jpg", "png", "jpeg", "pdf"],
            key="base_upload"
        )

        if base_file:
            with st.spinner("Processing base document..."):
                if base_file.type == "application/pdf":
                    base_img, total_pages = pdf_to_image(base_file)
                    if base_img:
                        st.success(f"✓ Loaded PDF (Page 1 of {total_pages})")
                        ss.base_image = base_img
                else:
                    base_img = Image.open(base_file)
                    if base_img.mode != 'RGB':
                        base_img = base_img.convert('RGB')
                    ss.base_image = base_img
                    st.success(f"✓ Loaded image: {base_file.name}")

    with col2:
        st.subheader("📋 Comparison Documents")
        comparison_files = st.file_uploader(
            "Upload images or PDFs to compare", 
            type=["jpg", "png", "jpeg", "pdf"],
            accept_multiple_files=True,
            key="comparison_upload"
        )

        if comparison_files:
            ss.comparison_images = []
            for file in comparison_files:
                with st.spinner(f"Processing {file.name}..."):
                    if file.type == "application/pdf":
                        comp_img, total_pages = pdf_to_image(file)
                        if comp_img:
                            ss.comparison_images.append((comp_img, f"{file.name} (Page 1)"))
                            st.success(f"✓ Loaded PDF: {file.name}")
                    else:
                        comp_img = Image.open(file)
                        if comp_img.mode != 'RGB':
                            comp_img = comp_img.convert('RGB')
                        ss.comparison_images.append((comp_img, file.name))
                        st.success(f"✓ Loaded image: {file.name}")

    if ss.base_image and ss.comparison_images:
        if st.button("🔍 Compare Documents", use_container_width=True, type="primary"):
            ss.results = []
            progress_bar = st.progress(0)
            total_comparisons = len(ss.comparison_images)

            for i, (comp_img, filename) in enumerate(ss.comparison_images):
                with st.spinner(f"Comparing with {filename}..."):
                    base_processed = preprocess_image(ss.base_image, enhance_contrast=enhance_contrast)
                    comp_processed = preprocess_image(comp_img, enhance_contrast=enhance_contrast)

                    if align_images_option:
                        comp_aligned, aligned_success = align_images(base_processed, comp_processed)
                        if not aligned_success:
                            comp_aligned = comp_processed
                    else:
                        comp_aligned = comp_processed

                    diff_results = find_differences(base_processed, comp_aligned, threshold, min_area)

                    # --- External ML Module Calls ---
                    base_symbols_raw = run_detection_pil(base_processed)
                    comp_symbols_raw = run_detection_pil(comp_aligned)
                    added, removed, misplaced = compare_labels(base_symbols_raw, comp_symbols_raw)

                    comp_classes = [d["class"] for d in comp_symbols_raw]
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
                            missing_box["label"] = "Missing"
                            comp_symbols_final.append(missing_box)
                            
                    for d in comp_symbols_raw:
                        if d["class"] not in [b["class"] for b in base_symbols_raw]:
                            added_box = d.copy()
                            added_box["label"] = "Added"
                            comp_symbols_final.append(added_box)
                            
                    comp_symbols = comp_symbols_final
                    base_symbols = []
                    for d in base_symbols_raw:
                        d = d.copy()
                        d["label"] = "Symbol"
                        base_symbols.append(d)
                        
                    base_text_boxes = filter_text_boxes(diff_results['bounding_boxes'], base_symbols)
                    comp_text_boxes = filter_text_boxes(diff_results['bounding_boxes'], comp_symbols)

                    base_marked = draw_differences(base_processed, base_text_boxes, color=(255,0,0))
                    base_marked = draw_symbol_boxes(base_marked, base_symbols, color_map={"Symbol": (0,255,0)})
                    comp_marked = draw_differences(comp_aligned, comp_text_boxes, color=(255,0,0))
                    comp_marked = draw_symbol_boxes(comp_marked, comp_symbols, color_map={"Added": (0,255,0), "Removed": (255,0,0), "Misplaced": (255,255,0)})

                    ss.results.append({
                        'filename': filename,
                        'base_marked': base_marked,
                        'comp_marked': comp_marked,
                        'difference_image': diff_results['difference_image'],
                        'ssim_score': diff_results['ssim_score'],
                        'total_differences': diff_results['total_differences'],
                        'bounding_boxes': diff_results['bounding_boxes'],
                        'base_symbols': base_symbols,
                        'comp_symbols': comp_symbols
                    })

                    progress_bar.progress((i + 1) / total_comparisons)

            st.success(f"✅ Comparison completed! Found differences in {len(ss.results)} documents.")

    if ss.results:
        st.markdown("---")
        st.header("📊 Comparison Results")

        tab_titles = [result['filename'] for result in ss.results]
        tabs = st.tabs(tab_titles)

        for tab, result in zip(tabs, ss.results):
            with tab:
                st.markdown(f"### 📄 {result['filename']}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SSIM Score", f"{result['ssim_score']:.4f}")
                with col2:
                    st.metric("Differences Found", result['total_differences'])
                with col3:
                    similarity_percent = result['ssim_score'] * 100
                    st.metric("Similarity %", f"{similarity_percent:.2f}%")

                st.subheader("Comparison View")
                colA, colB = st.columns(2)

                with colA:
                    st.markdown("**Base Document (OCR: red, Symbols: green)**")
                    st.image(result['base_marked'], use_column_width=True)
                    st.download_button(
                        "📥 Download Base with Differences",
                        image_to_bytes(result['base_marked']),
                        f"base_marked_{result['filename']}.png",
                        "image/png",
                        key=f"base_marked_{result['filename']}"
                    )

                with colB:
                    st.markdown(f"**{result['filename']} (OCR: red, Symbols: green)**")
                    st.image(result['comp_marked'], use_column_width=True)
                    st.download_button(
                        "📥 Download Comparison with Differences",
                        image_to_bytes(result['comp_marked']),
                        f"comparison_marked_{result['filename']}.png",
                        "image/png",
                        key=f"comp_marked_{result['filename']}"
                    )

if __name__ == "__main__":
    main()
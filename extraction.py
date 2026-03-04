import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import easyocr
import fitz  # PyMuPDF for PDF handling
import io
import torch

torch.classes.__path__ = []
st.set_page_config(layout="wide")

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
            font-weight:bold;font-size:18px;height:300px;
        }
        [data-testid=stWidgetLabel] {color:#f4a303;}
        .title-container {
            display:flex;justify-content:center;text-align:center;margin:auto;width:655px;
        }
        .image-display-container {
            background: rgba(244, 163, 3, 0.1);
            border: 2px solid #f4a303;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
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
        <span style="font-size:31px;font-weight:bold;color:#fff;">Welcome to</span>
        <img src="https://imgs.search.brave.com/v2uNTFRMI2DpvoTInWUOIV4tx2UtmaHH71Yg6fGLKoA/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9ub3Zp/bnRpeC5jb20vd3At/Y29udGVudC91cGxv/YWRzLzIwMjMvMTIv/TG9nby13aGl0ZS5w/bmc"
        style="height:43px;padding:2px;margin-bottom:10px;">
        <span style="font-size:31px;font-weight:bold;color:#f4a303;">OCR Tool</span>
    </div>
"""
with st.container(border=True):
    st.markdown(header_html, unsafe_allow_html=True)

# --- OCR Reader ---
@st.cache_resource
def init_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)
reader = init_ocr_reader()

# --- PDF Handler ---
def pdf_to_image(pdf_file, dpi=200):
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
        pdf_document.close()
        return image
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None

# --- Upload Handler ---
def process_uploaded_files(uploaded_files):
    processed = []
    for f in uploaded_files:
        try:
            if f.type == "application/pdf":
                img = pdf_to_image(f)
                if img:
                    processed.append((img, f"{f.name} (Page 1)"))
                    st.success(f"✓ Converted PDF: {f.name}")
            elif f.type in ["image/jpeg", "image/jpg", "image/png"]:
                img = Image.open(f)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                processed.append((img, f.name))
                st.success(f"✓ Loaded image: {f.name}")
            else:
                st.warning(f"⚠ Unsupported file type: {f.name}")
        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")
    return processed

# --- Image Filtering ---
def apply_filters(img, invert, grayscale, binary, denoise):
    if img is None:
        return None
    img = img.copy()
    if invert:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = ImageOps.invert(img)
    if grayscale:
        img = img.convert('L')
    if binary:
        if img.mode != 'L':
            img = img.convert('L')
        img = img.point(lambda x: 255 if x > 128 else 0, 'L')
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(3))
    return img

# --- OCR Execution ---
def run_ocr(img, contrast, adjust, text_ths, low_text, link_ths, canvas_sz, mag_ratio, reader):
    if img is None:
        return None, pd.DataFrame([["No image", ""]], columns=["Text","Confidence"])
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    draw_img = img.convert('RGB') if img.mode in ['L','1'] else img.copy()
    np_img = np.array(img)[:,:,::-1] if img.mode == 'RGB' else np.array(img).astype('uint8')
    try:
        results = reader.readtext(
            np_img,
            detail=1,
            contrast_ths=contrast,
            adjust_contrast=adjust,
            text_threshold=text_ths,
            low_text=low_text,
            link_threshold=link_ths,
            canvas_size=canvas_sz,
            mag_ratio=mag_ratio
        )
        draw = ImageDraw.Draw(draw_img)
        rows = []
        for coords, text, confidence in results:
            x_coords = [int(c[0]) for c in coords]
            y_coords = [int(c[1]) for c in coords]
            draw.polygon(list(zip(x_coords,y_coords)),outline='red',width=2)
            rows.append([text, f"{confidence:.2f}"])
        if not rows:
            rows.append(["No text detected", "0.00"])
        return draw_img, pd.DataFrame(rows, columns=["Text", "Confidence"])
    except Exception as e:
        st.error(f"OCR error: {e}")
        return draw_img, pd.DataFrame([["OCR Error", "0.00"]], columns=["Text", "Confidence"])

# --- File Conversion ---
def image_to_bytes(image):
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()

def dataframe_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Columnar DataFrame ---
def create_columnar_dataframe(all_results):
    max_rows = max(len(df) for _,df in all_results) if all_results else 0
    out = {'Row': list(range(1, max_rows+1))}
    for filename, df in all_results:
        texts = df['Text'].tolist()
        while len(texts) < max_rows:
            texts.append("")
        out[filename] = texts
    return pd.DataFrame(out)

def create_columnar_with_confidence(all_results):
    max_rows = max(len(df) for _,df in all_results) if all_results else 0
    out = {'Row': list(range(1, max_rows+1))}
    for filename, df in all_results:
        texts = df['Text'].tolist()
        confidences = df['Confidence'].tolist()
        while len(texts)<max_rows: texts.append("")
        while len(confidences)<max_rows: confidences.append("0.00")
        out[f"{filename}_Text"] = texts
        out[f"{filename}_Confidence"] = confidences
    return pd.DataFrame(out)

def export_to_columnar_excel(all_results, include_confidence=False):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            simple_df = create_columnar_dataframe(all_results)
            simple_df.to_excel(writer, sheet_name="OCR_Results", index=False)
            if include_confidence:
                detailed_df = create_columnar_with_confidence(all_results)
                detailed_df.to_excel(writer, sheet_name="Detailed_Results", index=False)
            summary = {
                'Filename': [fname for fname,_ in all_results],
                'Total_Texts': [len(df) for _,df in all_results],
                'Avg_Confidence': [df['Confidence'].astype(float).mean() for _,df in all_results]
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.error(f"Excel creation error: {e}")
        return None

# --- Main App ---
def main():
    ss = st.session_state
    if 'processed_images' not in ss:
        ss.processed_images = []
    if 'filtered_images' not in ss:
        ss.filtered_images = []
    if 'ocr_results' not in ss:
        ss.ocr_results = []
    if 'current_files' not in ss:
        ss.current_files = []
    if 'selected_image_index' not in ss:
        ss.selected_image_index = None


    col1, col2 = st.columns([1,1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Images or PDFs", type=["jpg", "png", "jpeg", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            cur_names = [f.name for f in uploaded_files]
            if cur_names != ss.current_files:
                ss.current_files = cur_names
                with st.spinner("Processing uploads..."):
                    ss.processed_images = process_uploaded_files(uploaded_files)
                    ss.filtered_images = []
                    ss.ocr_results = []
                    ss.selected_image_index = None  # Reset image display
            
    with col2:
        st.subheader("🔧 Preprocessing")
        c1,c2,c3,c4 = st.columns(4)
        with c1: invert = st.checkbox("Invert", value=False)
        with c2: gray = st.checkbox("Grayscale", value=False)
        with c3: binary = st.checkbox("Binary", value=False)
        with c4: denoise = st.checkbox("Denoise", value=False)
        if st.button("Apply Filters to All", use_container_width=True):
            with st.spinner("Filtering images..."):
                ss.filtered_images = []
                for img, fname in ss.processed_images:
                    filt = apply_filters(img, invert, gray, binary, denoise)
                    ss.filtered_images.append((filt, fname))
            st.success(f"Filtered {len(ss.filtered_images)} files.")
        if ss.filtered_images:
            st.subheader("⚙️ OCR Parameters")
            col1,col2,col3 = st.columns(3)
            with col1: contrast = st.slider("Contrast Threshold", 0., 1., 0.1)
            with col2: adjust = st.slider("Contrast Adjustment", 0., 1., 0.5)
            with col3: text_ths = st.slider("Text Threshold", 0., 1., 0.7)
            if st.button("🔍 Run OCR on All", use_container_width=True):
                ss.ocr_results = []
                ss.selected_image_index = None  # Reset image display
                pb = st.progress(0)
                total = len(ss.filtered_images)
                for i,(img,fname) in enumerate(ss.filtered_images):
                    with st.spinner(f"OCR for {fname}..."):
                        ocr_img, ocr_df = run_ocr(
                            np.array(img), contrast, adjust, text_ths, 0.4, 0.4, 2560, 1, reader
                        )
                        ss.ocr_results.append((fname, ocr_img, ocr_df))
                    pb.progress((i+1)/total)
                st.success(f"OCR done for {total} files!")

    # --- Enhanced OCR Results & Export with External Image Display ---
    if ss.ocr_results:
        st.subheader("📊 OCR Results")
        
        # External Image Display Area (outside tabs/columns)
        if ss.selected_image_index is not None:
            selected_fname, selected_ocr_img, _ = ss.ocr_results[ss.selected_image_index]
            
         
            header_col1, header_col2, header_col3 = st.columns([2, 6, 2])
           
            with header_col2:
                st.markdown(f"<h4 style='text-align:center;color:#f4a303;'>🖼️ {selected_fname}</h4>", unsafe_allow_html=True)
       
            with header_col3:
                if st.button("❌ Close Image", use_container_width=True, key="close_image_display"):
                    ss.selected_image_index = None
                    st.rerun()
            # Image display
            mid = st.columns([1,3,1])
            with mid[1]:
                st.image(selected_ocr_img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabs for file results
        tabs = st.tabs([
            f"📄 {fname[:20]}..." if len(fname)>20 else f"📄 {fname}"
            for fname,_,_ in ss.ocr_results
        ]) if len(ss.ocr_results)>1 else [st.container()]
        
        all_export = []
        
        for i,(fname,ocr_img,ocr_df) in enumerate(ss.ocr_results):
            with tabs[i]:
                st.write(f"**File:** {fname}")
                col1,col2 = st.columns(2)
                
                with col1:
                    # Button to display image externally
                    button_text = "🖼️ View OCR Image" if ss.selected_image_index != i else "🔄 Refresh View"
                    if st.button(button_text, use_container_width=True, key=f"view_img_{i}"):
                        ss.selected_image_index = i
                        st.rerun()
                        
                with col2: 
                    st.download_button(
                        "📥 Download OCR Image",
                        image_to_bytes(ocr_img), f"ocr_{fname}.png", "image/png", 
                        key=f"img_dl_{i}", use_container_width=True
                    )
                      
            all_export.append((fname, ocr_df))
            
        # Combined results section
        st.subheader("Extracted contents")
        col_df = create_columnar_dataframe(all_export)
        st.dataframe(col_df, use_container_width=True, hide_index=True)

        excel_bytes = export_to_columnar_excel(all_export, include_confidence=False)
        if excel_bytes:
            st.download_button(
                "📥 Download Excel",
                excel_bytes, "ocr_results_simple.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()

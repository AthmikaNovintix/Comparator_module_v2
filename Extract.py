import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import easyocr
import zxingcpp
from PIL import Image

# --- ADD THESE TWO LINES TO PREVENT CLOUD CRASHES ---
import torch
torch.set_num_threads(1) 

# Import your existing YOLO logic
from detect import run_detection_pil

# -----------------------------
# Lazy Load OCR (Prevents RAM crashes)
# -----------------------------
@st.cache_resource
def get_ocr_reader():
    """Loads EasyOCR only when needed and keeps it in memory"""
    return easyocr.Reader(['en', 'fr', 'de'], gpu=False)

def extract_barcodes(image):
    """Decodes barcodes using ZXing to get perfect GTINs (Avoids Windows DLL issues)"""
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Convert RGB to BGR for standard OpenCV/ZXing processing
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
    barcodes = zxingcpp.read_barcodes(img_np)
    return [bc.text for bc in barcodes]

def detect_logos(image, logo_folder="logos"):
    """Compares the label against a folder of known logos using SIFT safely"""
    if not os.path.exists(logo_folder):
        return []

    label_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(label_gray, None)
    
    # If the main image has no features, we can't match anything
    if des1 is None or len(des1) < 2:
        return []
    
    detected_logos = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for logo_file in os.listdir(logo_folder):
        # 1. Skip non-image files (like Thumbs.db or .DS_Store)
        if not logo_file.lower().endswith(valid_extensions):
            continue
            
        logo_path = os.path.join(logo_folder, logo_file)
        
        try:
            logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
            if logo_img is None: 
                continue
                
            kp2, des2 = sift.detectAndCompute(logo_img, None)
            
            # 2. Skip if the logo is too simple/small and generates no descriptors
            if des2 is None or len(des2) < 2:
                continue
            
            # Feature matching using FLANN
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des2, des1, k=2)
            
            # STRICTER Lowe's ratio test to find good matches safely (0.6 instead of 0.7)
            good_matches = []
            for match_group in matches:
                if len(match_group) == 2:
                    m, n = match_group
                    if m.distance < 0.6 * n.distance:
                        good_matches.append(m)
            
            # STRICTER Threshold: Require 30 matches AND 10% of logo features to stop hallucinations
            MIN_MATCH_COUNT = 30
            if len(good_matches) > MIN_MATCH_COUNT and len(good_matches) > (0.10 * len(kp2)): 
                detected_logos.append(logo_file.rsplit('.', 1)[0])
                
        except Exception as e:
            # Catch any unexpected OpenCV/YOLO errors silently so the loop continues
            print(f"Skipping {logo_file} due to error: {e}")
            continue
            
    return detected_logos

def extract_all_features(image, logo_folder="logos"):
    """Master function to extract Text, Symbols, Barcodes, and Logos"""
    features = []

    # 1. Barcode Extraction (100% accurate GTIN)
    barcodes = extract_barcodes(image)
    for bc in barcodes:
        features.append({"Type": "Barcode", "Value": bc})

    # 2. Logo / Image Extraction
    logos = detect_logos(image, logo_folder)
    for logo in logos:
        features.append({"Type": "Image", "Value": f"Image - {logo}"})

    # 3. Symbol Extraction (Using your YOLO models)
    symbols = run_detection_pil(image)
    for sym in symbols:
        features.append({"Type": "Symbol", "Value": sym["class"]})

    # 4. Text Extraction 
    reader = get_ocr_reader() # <--- CALL THE LAZY LOADED OCR HERE
    np_img = np.array(image)
    ocr_results = reader.readtext(
        np_img, 
        detail=0,          # Return just the text
        mag_ratio=1.5,     # Magnify image to catch small German/French text
        contrast_ths=0.1,  # Lower threshold for faint text
        text_threshold=0.6 # Strictness
    )
    
    # Filter out text that perfectly matches the barcode to avoid duplicates
    for text in ocr_results:
        clean_text = text.strip()
        if clean_text and clean_text not in barcodes:
            features.append({"Type": "Text", "Value": clean_text})

    return pd.DataFrame(features)

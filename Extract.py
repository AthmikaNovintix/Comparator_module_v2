import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import easyocr
import zxingcpp
from PIL import Image
import gc  # NEW: Python Garbage Collector

# --- Prevent PyTorch from using too much RAM ---
import torch
torch.set_num_threads(1) 

# Import your existing YOLO logic
from detect import run_detection_pil

def extract_barcodes(image):
    """Decodes barcodes using ZXing to get perfect GTINs"""
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    barcodes = zxingcpp.read_barcodes(img_np)
    return [bc.text for bc in barcodes]

def detect_logos(image, logo_folder="logos"):
    """Compares the label against a folder of known logos safely"""
    if not os.path.exists(logo_folder):
        return []

    label_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(label_gray, None)
    
    if des1 is None or len(des1) < 2:
        return []
    
    detected_logos = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for logo_file in os.listdir(logo_folder):
        if not logo_file.lower().endswith(valid_extensions):
            continue
            
        logo_path = os.path.join(logo_folder, logo_file)
        
        try:
            logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
            if logo_img is None: continue
                
            kp2, des2 = sift.detectAndCompute(logo_img, None)
            if des2 is None or len(des2) < 2: continue
            
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des2, des1, k=2)
            
            good_matches = []
            for match_group in matches:
                if len(match_group) == 2:
                    m, n = match_group
                    if m.distance < 0.6 * n.distance:
                        good_matches.append(m)
            
            MIN_MATCH_COUNT = 30
            if len(good_matches) > MIN_MATCH_COUNT and len(good_matches) > (0.10 * len(kp2)): 
                detected_logos.append(logo_file.rsplit('.', 1)[0])
                
        except Exception as e:
            print(f"Skipping {logo_file} due to error: {e}")
            continue
            
    return detected_logos

def extract_all_features(image, logo_folder="logos"):
    """Master function to extract Text, Symbols, Barcodes, and Logos"""
    features = []

    # 1. Barcode Extraction
    barcodes = extract_barcodes(image)
    for bc in barcodes:
        features.append({"Type": "Barcode", "Value": bc})

    # 2. Logo / Image Extraction
    logos = detect_logos(image, logo_folder)
    for logo in logos:
        features.append({"Type": "Image", "Value": f"Image - {logo}"})

    # 3. Symbol Extraction (YOLO loads, runs, and deletes itself inside here)
    symbols = run_detection_pil(image)
    for sym in symbols:
        features.append({"Type": "Symbol", "Value": sym["class"]})

    # 4. Text Extraction (Load and Dump OCR)
    reader = easyocr.Reader(['en', 'fr', 'de'], gpu=False)
    
    np_img = np.array(image)
    ocr_results = reader.readtext(
        np_img, 
        detail=0,          
        mag_ratio=1.5,     
        contrast_ths=0.1,  
        text_threshold=0.6 
    )
    
    for text in ocr_results:
        clean_text = text.strip()
        if clean_text and clean_text not in barcodes:
            features.append({"Type": "Text", "Value": clean_text})

    # CRITICAL: Delete OCR and force RAM cleanup
    del reader
    gc.collect()

    return pd.DataFrame(features)

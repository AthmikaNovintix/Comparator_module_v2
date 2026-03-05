import pandas as pd
import numpy as np
import cv2
import os
import zxingcpp
import pytesseract
from PIL import Image

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

def extract_all_features(image, precomputed_symbols, logo_folder="logos"):
    """Master function to extract Text, Barcodes, Logos, and append Symbols"""
    features = []

    # 1. Barcode Extraction
    barcodes = extract_barcodes(image)
    for bc in barcodes:
        features.append({"Type": "Barcode", "Value": bc})

    # 2. Logo / Image Extraction
    logos = detect_logos(image, logo_folder)
    for logo in logos:
        features.append({"Type": "Image", "Value": f"Image - {logo}"})

    # 3. Append Symbols (Passed from app.py to save massive amounts of RAM!)
    for sym in precomputed_symbols:
        features.append({"Type": "Symbol", "Value": sym["class"]})

    # 4. Ultra-lightweight Text Extraction (Tesseract)
    np_img = np.array(image)
    if len(np_img.shape) == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img
        
    # Scale image up slightly to catch tiny German/French text
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Run OCR for English, French, and German using practically zero RAM
    ocr_text = pytesseract.image_to_string(gray, lang='eng+fra+deu')
    
    for line in ocr_text.split('\n'):
        clean_text = line.strip()
        # Ignore empty lines, tiny noise specks, or duplicate barcodes
        if len(clean_text) > 2 and clean_text not in barcodes:
            features.append({"Type": "Text", "Value": clean_text})

    return pd.DataFrame(features)

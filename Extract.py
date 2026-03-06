import pandas as pd
import numpy as np
import cv2
import os
import zxingcpp
import pytesseract
import re
from PIL import Image
import gc

# --- Prevent PyTorch/Tesseract from using too much RAM ---
import torch
torch.set_num_threads(1) 

from detect import run_detection_pil

# ---------------------------------------------------------
# ADVANCED BARCODE SCANNER LOGIC (From agentic_scanner.py)
# ---------------------------------------------------------
def _zxing_decode(image):
    """Internal helper to run ZXing with advanced parameters"""
    found = []
    try:
        zx_results = zxingcpp.read_barcodes(
            image, 
            try_rotate=True, 
            try_invert=True,
            try_downscale=True
        )
        for res in zx_results:
            # ZXing Python bindings return objects with .text and .format properties
            found.append({"data": res.text, "type": res.format.name})
    except Exception as e:
        print(f"Error in zxing decode: {e}")
    return found

def extract_barcodes(image):
    """Master Barcode Extractor with Agentic Fallback & Preprocessing"""
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    results = []

    # 1. Standard Decode
    results.extend(_zxing_decode(img_np))

    # 2. Channel-based Preprocessing
    if len(results) < 2 and len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Blue channel often neutralizes blue/light-blue watermarks
        blue = img_np[:, :, 0]
        results.extend(_zxing_decode(cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)))
        
        # Histogram Equalization
        equ = cv2.equalizeHist(blue)
        results.extend(_zxing_decode(cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)))

    # 3. Agentic Scanline Averaging Pass (Fallback if Code128 is missing)
    has_code128 = any(r['type'] == 'Code128' for r in results)
    if not has_code128:
        h, w = img_np.shape[:2]
        
        # ROI: Bottom 30% of the image
        roi = img_np[int(h*0.7):h, :]
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
            
        # Scanline Averaging: Average pixels vertically to eliminate noise/watermarks
        avg_line = np.mean(gray, axis=0).astype(np.uint8)
        
        # Re-broadcast the 1D line back to 2D for zxing
        scan_img = np.tile(avg_line, (200, 1))
        
        # Add quiet zone padding
        padded = cv2.copyMakeBorder(scan_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
        processed = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        
        # Upscale 2x for better resolution
        h_p, w_p = processed.shape[:2]
        processed = cv2.resize(processed, (w_p*2, h_p), interpolation=cv2.INTER_CUBIC)
        
        results.extend(_zxing_decode(processed))

    # 4. De-duplicate results
    unique_barcodes = []
    seen = set()
    for res in results:
        data = res['data']
        if data not in seen:
            seen.add(data)
            unique_barcodes.append(data)
            
    return unique_barcodes

# ---------------------------------------------------------
# EXISTING LOGO, TEXT, AND MASTER EXTRACTION LOGIC
# ---------------------------------------------------------
def detect_logos(image, logo_folder="logos"):
    """Compares the label against a folder of known logos using SIFT + RANSAC Geometry Verification"""
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
                    # Relaxed distance threshold for text-heavy logos
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # ---------------------------------------------------------
            # THE FIX: RANSAC GEOMETRY VERIFICATION
            # ---------------------------------------------------------
            # Threshold lowered to 10 to catch small/text logos safely
            if len(good_matches) >= 10: 
                src_pts = np.float32([ kp2[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                
                # RANSAC ensures the matching points actually form the physical shape of the logo
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    inliers = np.sum(mask)
                    # If points form a valid geometric shape, it's a confirmed logo
                    if inliers >= 10 and inliers >= (0.05 * len(kp2)):
                        detected_logos.append(logo_file.rsplit('.', 1)[0])
                        
        except Exception as e:
            print(f"Skipping {logo_file} due to error: {e}")
            continue
            
    # Return unique values only to prevent duplicates
    return list(set(detected_logos)) 

def extract_all_features(image, precomputed_symbols, logo_folder="logos"):
    """Master function to extract Text, Barcodes, Logos, and append Symbols"""
    features = []

    # 1. Advanced Barcode Extraction (Includes Agentic Scanner Logic)
    barcodes = extract_barcodes(image)
    for bc in barcodes:
        features.append({"Type": "Barcode", "Value": bc})

    # 2. Logo / Image Extraction (Now powered by RANSAC)
    logos = detect_logos(image, logo_folder)
    for logo in logos:
        features.append({"Type": "Image", "Value": f"Image - {logo}"})

    # 3. Append Precomputed YOLO Symbols
    for sym in precomputed_symbols:
        features.append({"Type": "Symbol", "Value": sym["class"]})

    # 4. Ultra-lightweight Text Extraction (Tesseract)
    np_img = np.array(image)
    if len(np_img.shape) == 3 and np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img
        
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    ocr_text = pytesseract.image_to_string(gray, lang='eng+fra+deu')
    
    for line in ocr_text.split('\n'):
        clean_text = re.sub(r'[|><_~=«»"]', '', line).strip()
        if len(clean_text) > 2 and any(c.isalnum() for c in clean_text) and clean_text not in barcodes:
            features.append({"Type": "Text", "Value": clean_text})

    return pd.DataFrame(features)

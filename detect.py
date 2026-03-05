import streamlit as st
from ultralytics import YOLO
import cv2
import math
import tempfile
import gc  # NEW: Python Garbage Collector

# -----------------------------
# Run Detection (For Streamlit/PIL)
# -----------------------------
def run_detection_pil(pil_image):
    """Handles PIL images using a Load-and-Dump memory approach"""
    # 1. Load models into RAM
    model16 = YOLO("16sym_models/best.pt")
    model4 = YOLO("4sym_models/best.pt")
    
    # 2. Process the image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        results16 = model16(tmp.name, conf=0.3)[0]
        results4 = model4(tmp.name, conf=0.3)[0]
        
    detections = []
    for box in results16.boxes:
        cls = model16.names[int(box.cls)]
        bbox = box.xyxy[0].tolist()
        detections.append({"class": cls, "bbox": bbox, "label": "Symbol"})
        
    for box in results4.boxes:
        cls = model4.names[int(box.cls)]
        bbox = box.xyxy[0].tolist()
        detections.append({"class": cls, "bbox": bbox, "label": "Symbol"})
        
    # 3. CRITICAL: Delete models and force RAM cleanup
    del model16
    del model4
    gc.collect()
        
    return detections

# -----------------------------
# Utility / Math Logic
# -----------------------------
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compare_labels(base_det, edited_det, threshold=40):
    added = []
    removed = []
    misplaced = []
    
    base_classes = [d["class"] for d in base_det]
    edited_classes = [d["class"] for d in edited_det]
    
    for d in edited_det:
        if d["class"] not in base_classes:
            d = d.copy()
            d["label"] = "Added"
            added.append(d)
            
    for d in base_det:
        if d["class"] not in edited_classes:
            d = d.copy()
            d["label"] = "Removed"
            removed.append(d)
            
    for b in base_det:
        for e in edited_det:
            if b["class"] == e["class"]:
                c1 = get_center(b["bbox"])
                c2 = get_center(e["bbox"])
                dist = math.dist(c1, c2)
                if dist > threshold:
                    e = e.copy()
                    e["label"] = "Misplaced"
                    misplaced.append(e)
                    
    return added, removed, misplaced

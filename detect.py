import streamlit as st
from ultralytics import YOLO
import cv2
import math
import tempfile

# -----------------------------
# Lazy Load Models (Prevents RAM crashes)
# -----------------------------
@st.cache_resource
def load_yolo_models():
    """Loads YOLO models only when needed and keeps them in memory"""
    m16 = YOLO("16sym_models/best.pt")
    m4 = YOLO("4sym_models/best.pt")
    return m16, m4

# -----------------------------
# Run Detection (For File Paths)
# -----------------------------
def run_detection(image_path):
    model16, model4 = load_yolo_models() # Get the cached models
    
    results16 = model16(image_path, conf=0.3)[0]
    results4 = model4(image_path, conf=0.3)[0]

    detections = []

    for box in results16.boxes:
        cls = model16.names[int(box.cls)]
        bbox = box.xyxy[0].tolist()
        detections.append({"class": cls, "bbox": bbox})

    for box in results4.boxes:
        cls = model4.names[int(box.cls)]
        bbox = box.xyxy[0].tolist()
        detections.append({"class": cls, "bbox": bbox})

    return detections

# -----------------------------
# Run Detection (For Streamlit/PIL)
# -----------------------------
def run_detection_pil(pil_image):
    """Handles PIL images coming from Streamlit"""
    model16, model4 = load_yolo_models() # Get the cached models
    
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

# -----------------------------
# CLI Output Visualization
# -----------------------------
def draw_results(image_path, added, removed, misplaced):
    img = cv2.imread(image_path)

    for d in added:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, "Added", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    for d in removed:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(img, "Removed", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    for d in misplaced:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(img, "Misplaced", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imwrite("comparison_output.jpg", img)
    print("✅ Output saved as comparison_output.jpg")

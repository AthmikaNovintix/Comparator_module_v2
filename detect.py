from ultralytics import YOLO
import cv2
import math
import tempfile

# -----------------------------
# Load Models
# -----------------------------
# Models are loaded once when this module is imported.
model16 = YOLO("16sym_models/best.pt")
model4 = YOLO("4sym_models/best.pt")


# -----------------------------
# Run Detection (For File Paths)
# -----------------------------
def run_detection(image_path):
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

    # Green → Added
    for d in added:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, "Added", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Red → Removed
    for d in removed:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(img, "Removed", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Yellow → Misplaced
    for d in misplaced:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(img, "Misplaced", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imwrite("comparison_output.jpg", img)
    print("✅ Output saved as comparison_output.jpg")


# -----------------------------
# MAIN (For local CLI testing)
# -----------------------------
if __name__ == "__main__":
    base_image = "LCN_IMG_Base.jpg"
    edited_image = "LCN-IMG_CHILD_sym_removed.jpg"

    print("🔍 Running detection...")
    base_det = run_detection(base_image)
    edited_det = run_detection(edited_image)

    print("📊 Comparing labels...")
    added, removed, misplaced = compare_labels(base_det, edited_det)

    print(f"Added: {len(added)}")
    print(f"Removed: {len(removed)}")
    print(f"Misplaced: {len(misplaced)}")

    draw_results(edited_image, added, removed, misplaced)
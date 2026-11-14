from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
import gc
import json
import torch
import numpy as np
from PIL import Image

torch.set_num_threads(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = YOLO("best.pt")
    dummy_img = Image.new("RGB", (480, 480))
    with torch.no_grad():
        model.predict(dummy_img, imgsz=480, verbose=False)
    del dummy_img
    gc.collect()
    print("✅ โหลดและวอร์มโมเดลแพะ (best.pt) พร้อมใช้งาน!")
except Exception as e:
    print(f"⚠️ โหลด best.pt ไม่สำเร็จ ({e}) -> ใช้ yolov8n.pt แทน")
    model = YOLO("yolov8n.pt")


def get_overlap_ratio(goat_box, human_box):
    gx1, gy1, gx2, gy2 = goat_box
    hx1, hy1, hx2, hy2 = human_box

    inter_x1 = max(gx1, hx1)
    inter_y1 = max(gy1, hy1)
    inter_x2 = min(gx2, hx2)
    inter_y2 = min(gy2, hy2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    goat_area = max((gx2 - gx1) * (gy2 - gy1), 1.0)
    return inter_area / goat_area


def detect_valid_goats(image: Image.Image):
    with torch.no_grad():
        results = model.predict(
            image,
            conf=0.16,
            iou=0.50,
            imgsz=480,
            agnostic_nms=False,
            verbose=False
        )

    result = results[0]
    raw_goats = []
    human_boxes = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = box.conf.item()
        coords = [round(v) for v in box.xyxy[0].tolist()]

        if class_name.lower() == "human" and confidence >= 0.25:
            human_boxes.append((coords, confidence))
        elif class_name.lower() == "goat":
            raw_goats.append({
                "box": coords,
                "confidence": round(confidence, 2),
                "raw_conf": confidence,
                "class_id": class_id,
                "class_name": class_name
            })

    valid_goats = []
    for g in raw_goats:
        is_false_on_human = False
        for h_box, h_conf in human_boxes:
            overlap = get_overlap_ratio(g["box"], h_box)
            if overlap > 0.65 and h_conf >= g["raw_conf"]:
                is_false_on_human = True
                break
        if not is_false_on_human:
            valid_goats.append(g)

    del results
    return valid_goats


def extract_goat_vector(crop_img: Image.Image) -> list:
    # 1. สกัดลักษณะลวดลายสีขนแบบแบ่งโซน (Spatial HSV Color Pattern)
    resized = crop_img.resize((128, 128)).convert("HSV")
    arr = np.array(resized, dtype=np.float32)

    regions = [
        arr[0:64, 0:64],
        arr[0:64, 64:128],
        arr[64:128, 0:64],
        arr[64:128, 64:128],
        arr[24:104, 24:104],
    ]

    color_features = []
    for reg in regions:
        h_hist, _ = np.histogram(reg[:, :, 0], bins=16, range=(0, 256))
        s_hist, _ = np.histogram(reg[:, :, 1], bins=8, range=(0, 256))
        v_hist, _ = np.histogram(reg[:, :, 2], bins=8, range=(0, 256))
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        color_features.append(hist)

    color_vec = np.concatenate(color_features)
    color_vec /= (np.linalg.norm(color_vec) + 1e-8)

    # 2. ดึง Deep Feature จากโครงสร้างรูปร่างของโมเดล YOLOv8
    try:
        with torch.no_grad():
            embeds = model.embed(crop_img, imgsz=224, verbose=False)
        deep_vec = embeds[0].detach().cpu().numpy().flatten().astype(np.float32)
        deep_vec /= (np.linalg.norm(deep_vec) + 1e-8)
        combined = np.concatenate([deep_vec * 0.6, color_vec * 0.4])
    except Exception:
        combined = color_vec

    combined /= (np.linalg.norm(combined) + 1e-8)
    return [round(float(x), 5) for x in combined]


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


@app.get("/")
def read_root():
    return {"status": "AI Server is online and ready!"}


@app.post("/count-goat")
async def count_goat(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        valid_goats = detect_valid_goats(image)

        detections = [
            {
                "box": g["box"],
                "confidence": g["confidence"],
                "class_id": g["class_id"],
                "class_name": g["class_name"]
            }
            for g in valid_goats
        ]

        del image, contents
        gc.collect()

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "message": f"Found {len(detections)} goats"
        }
    except Exception as e:
        gc.collect()
        return {"success": False, "count": 0, "error": str(e)}


@app.post("/extract-goat-feature")
async def extract_goat_feature(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = image.size

        valid_goats = detect_valid_goats(image)
        goat_detected = len(valid_goats) > 0

        if goat_detected:
            best_goat = max(
                valid_goats,
                key=lambda g: (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1])
            )
            x1, y1, x2, y2 = best_goat["box"]
            crop_img = image.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        else:
            crop_img = image.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))

        embedding = extract_goat_vector(crop_img)

        del image, crop_img, contents
        gc.collect()

        return {
            "success": True,
            "goat_detected": goat_detected,
            "embedding": embedding
        }
    except Exception as e:
        gc.collect()
        return {"success": False, "error": str(e)}


@app.post("/identify-goat")
async def identify_goat(
    file: UploadFile = File(...),
    registered_goats: str = Form("[]")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = image.size

        db_goats = json.loads(registered_goats)
        valid_goats = detect_valid_goats(image)
        detections = []

        for g in valid_goats:
            x1, y1, x2, y2 = g["box"]
            crop_img = image.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            query_vec = extract_goat_vector(crop_img)

            best_match = None
            best_score = -1.0

            for reg in db_goats:
                for emb_obj in reg.get("embeddings", []):
                    ref_vec = emb_obj.get("vector", [])
                    score = cosine_similarity(query_vec, ref_vec)
                    if score > best_score:
                        best_score = score
                        best_match = reg

            if best_match and best_score >= 0.72:
                identity_info = {
                    "is_registered": True,
                    "goat_id": best_match.get("id", ""),
                    "earTag": best_match.get("earTag", ""),
                    "name": best_match.get("name", ""),
                    "similarity": round(best_score * 100, 1)
                }
            else:
                identity_info = {
                    "is_registered": False,
                    "name": "แพะใหม่ (ยังไม่ลงทะเบียน)",
                    "similarity": round(max(best_score, 0.0) * 100, 1)
                }

            detections.append({
                "box": g["box"],
                "confidence": g["confidence"],
                "class_id": g["class_id"],
                "class_name": g["class_name"],
                "identity": identity_info
            })

            del crop_img

        del image, contents
        gc.collect()

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }
    except Exception as e:
        gc.collect()
        return {"success": False, "count": 0, "error": str(e)}

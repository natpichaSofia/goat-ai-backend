from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
import gc
import json
import torch
import numpy as np
from PIL import Image, ImageOps

torch.set_num_threads(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

IS_CUSTOM_GOAT_MODEL = False

try:
    model = YOLO("best.pt")
    dummy_img = Image.new("RGB", (320, 320))
    with torch.no_grad():
        model.predict(dummy_img, imgsz=320, verbose=False)
    del dummy_img
    gc.collect()
    names_lower = [str(v).lower() for v in model.names.values()]
    IS_CUSTOM_GOAT_MODEL = any(k in names_lower for k in ["goat", "goats", "แพะ"])
    print(f"✅ โหลดโมเดล best.pt สำเร็จ! names={model.names}")
except Exception as e:
    print(f"⚠️ โหลด best.pt ไม่สำเร็จ ({e}) -> ใช้ yolov8n.pt แทน")
    model = YOLO("yolov8n.pt")
    IS_CUSTOM_GOAT_MODEL = False


def get_box_metrics(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0, 0.0, 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = max(float((ax2 - ax1) * (ay2 - ay1)), 1.0)
    area_b = max(float((bx2 - bx1) * (by2 - by1)), 1.0)
    union_area = area_a + area_b - inter_area

    iou = inter_area / max(union_area, 1.0)
    containment = inter_area / min(area_a, area_b)
    return iou, containment, area_a / area_b


def detect_valid_goats(image: Image.Image):
    w, h = image.size
    img_area = max(float(w * h), 1.0)

    with torch.no_grad():
        results = model.predict(
            image,
            conf=0.10,
            iou=0.45,
            imgsz=480,
            agnostic_nms=False,
            verbose=False
        )

    result = results[0]
    raw_goats = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(model.names.get(class_id, "")).lower()
        confidence = box.conf.item()
        coords = [round(v) for v in box.xyxy[0].tolist()]

        box_w = max(coords[2] - coords[0], 0)
        box_h = max(coords[3] - coords[1], 0)
        box_area = float(box_w * box_h)

        if (box_area / img_area) < 0.018:
            continue

        is_goat_class = False
        if IS_CUSTOM_GOAT_MODEL:
            if class_name in ["goat", "goats", "แพะ"] or (
                class_id == 0 and class_name not in ["human", "person"]
            ):
                is_goat_class = True
        else:
            if class_name in ["goat", "sheep", "cow", "dog"] or class_id in [16, 18, 19]:
                is_goat_class = True

        if is_goat_class:
            raw_goats.append({
                "box": coords,
                "confidence": round(confidence, 2),
                "raw_conf": confidence,
                "area": box_area,
                "class_id": 0,
                "class_name": "goat"
            })

    # ยุบกรอบซ้อนและกรอบชิ้นส่วนย่อยให้เหลือ 1 กรอบเต็มตัวต่อแพะ 1 ตัว
    raw_goats.sort(key=lambda item: item["raw_conf"] * (item["area"] ** 0.35), reverse=True)
    deduped_goats = []

    for candidate in raw_goats:
        is_duplicate = False
        for idx, kept in enumerate(deduped_goats):
            iou, containment, area_ratio = get_box_metrics(candidate["box"], kept["box"])
            if iou > 0.35 or containment > 0.55:
                if area_ratio > 1.35 and candidate["raw_conf"] >= (kept["raw_conf"] * 0.65):
                    deduped_goats[idx] = candidate
                is_duplicate = True
                break
        if not is_duplicate:
            deduped_goats.append(candidate)

    del results
    return deduped_goats


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    vec = vec - np.mean(vec)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def compute_deep_feature(crop_img: Image.Image):
    try:
        with torch.no_grad():
            embeds = model.embed(crop_img, imgsz=160, verbose=False)
        raw_deep = embeds[0].detach().cpu().numpy().flatten().astype(np.float32)
        signed_sqrt = np.sign(raw_deep) * np.sqrt(np.abs(raw_deep))
        return normalize_vec(signed_sqrt)
    except Exception:
        return None


def extract_goat_vector(crop_img: Image.Image, deep_vec=None) -> list:
    resized_rgb = crop_img.resize((128, 128))
    hsv_arr = np.array(resized_rgb.convert("HSV"), dtype=np.float32)
    lab_arr = np.array(resized_rgb.convert("LAB"), dtype=np.float32)
    gray_arr = np.array(resized_rgb.convert("L"), dtype=np.float32)

    anatomical_zones = [
        (slice(0, 56), slice(16, 112), 1.8),    # ส่วนหัว ใบหน้า เขา หู
        (slice(24, 72), slice(20, 108), 1.6),   # ลำคอ และหน้าอก (ปลอกคอ/เครา)
        (slice(20, 62), slice(18, 110), 1.4),   # แผ่นหลังและลำตัวช่วงบน
        (slice(36, 82), slice(14, 64), 1.0),    # ลำตัวซีกซ้าย
        (slice(36, 82), slice(64, 114), 1.0),   # ลำตัวซีกขวา
    ]

    color_features = []
    for sy, sx, zone_weight in anatomical_zones:
        h_reg = hsv_arr[sy, sx]
        l_reg = lab_arr[sy, sx]

        h_hist, _ = np.histogram(h_reg[:, :, 0].flatten(), bins=16, range=(0, 256))
        s_hist, _ = np.histogram(h_reg[:, :, 1].flatten(), bins=12, range=(0, 256))
        l_hist, _ = np.histogram(l_reg[:, :, 0].flatten(), bins=16, range=(0, 256))
        a_hist, _ = np.histogram(l_reg[:, :, 1].flatten(), bins=12, range=(0, 256))
        b_hist, _ = np.histogram(l_reg[:, :, 2].flatten(), bins=12, range=(0, 256))

        hist = np.concatenate([h_hist, s_hist, l_hist, a_hist, b_hist]).astype(np.float32)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist /= hist_sum
        hist = np.sqrt(np.maximum(hist, 0.0))
        hist = normalize_vec(hist) * zone_weight
        color_features.append(hist)

    appearance_vec = normalize_vec(np.concatenate(color_features))

    body_gray = gray_arr[8:104, 12:116]
    gx = body_gray[:, 1:] - body_gray[:, :-1]
    gy = body_gray[1:, :] - body_gray[:-1, :]
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

    edge_features = []
    cell_h, cell_w = 23, 25
    for r in range(4):
        for c in range(4):
            sub_mag = mag[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w].flatten()
            sub_ang = ang[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w].flatten()
            e_hist, _ = np.histogram(sub_ang, bins=8, range=(0, 180), weights=sub_mag)
            e_sum = np.sum(e_hist)
            if e_sum > 0:
                e_hist = e_hist / e_sum
            edge_features.append(normalize_vec(np.sqrt(np.maximum(e_hist, 0.0))))

    edge_vec = normalize_vec(np.concatenate(edge_features))

    if deep_vec is not None:
        combined = np.concatenate([
            appearance_vec * 0.55,
            deep_vec * 0.30,
            edge_vec * 0.15
        ])
    else:
        combined = np.concatenate([appearance_vec * 0.75, edge_vec * 0.25])

    combined = normalize_vec(combined)
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
    return {
        "status": "AI Server is online and ready!",
        "custom_model": IS_CUSTOM_GOAT_MODEL,
        "classes": model.names
    }


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
            best_goat = max(valid_goats, key=lambda g: g["area"])
            x1, y1, x2, y2 = best_goat["box"]
            crop_img = image.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        else:
            crop_img = image.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))

        deep_vec = compute_deep_feature(crop_img)
        embedding = extract_goat_vector(crop_img, deep_vec)

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
            cw, ch = crop_img.size

            # คำนวณ deep_vec เพียง 1 ครั้งต่อตัว แล้วแชร์ให้มุมกลับด้านและมุมตัดชิด
            deep_vec = compute_deep_feature(crop_img)
            query_vecs = [
                extract_goat_vector(crop_img, deep_vec),
                extract_goat_vector(ImageOps.mirror(crop_img), deep_vec),
            ]
            if cw > 40 and ch > 40:
                tight_crop = crop_img.crop((int(cw * 0.05), int(ch * 0.05), int(cw * 0.95), int(ch * 0.95)))
                query_vecs.append(extract_goat_vector(tight_crop, deep_vec))

            best_match = None
            best_score = -1.0

            for reg in db_goats:
                for emb_obj in reg.get("embeddings", []):
                    ref_vec = emb_obj.get("vector", [])
                    for q_vec in query_vecs:
                        score = cosine_similarity(q_vec, ref_vec)
                        if score > best_score:
                            best_score = score
                            best_match = reg

            if best_match and best_score >= 0.78:
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
                    "name": "แพะ (ยังไม่ลงทะเบียน)",
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

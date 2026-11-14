from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
import gc
import torch
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
    # คำนวณสัดส่วนพื้นที่ของกรอบแพะที่ซ้อนทับอยู่ภายในกรอบคน (0.0 - 1.0)
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

@app.get("/")
def read_root():
    return {"status": "AI Server is online and ready!"}

@app.post("/count-goat")
async def count_goat(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # ปิด agnostic_nms เพื่อไม่ให้กรอบ Human ลบกรอบ Goat ที่ยืนอยู่ข้างๆ
        # เพิ่ม imgsz=480 ให้เห็นรายละเอียดหน้าแพะชัดขึ้น และลด conf=0.16 เพื่อเก็บแพะตัวจริงให้ครบ
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

        # 1. แยกเก็บกรอบ Human และกรอบ Goat ออกมาก่อน
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()
            coords = [round(v) for v in box.xyxy[0].tolist()]

            print(f"🔍 เจอวัตถุ: {class_name} (มั่นใจ {round(confidence * 100, 1)}%)")

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

        # 2. กรองเฉพาะกรอบ Goat ที่ไม่ได้ซ้อนทับอยู่บนตัวคนเกิน 65%
        detections = []
        for g in raw_goats:
            is_false_on_human = False
            for h_box, h_conf in human_boxes:
                overlap = get_overlap_ratio(g["box"], h_box)
                if overlap > 0.65 and h_conf >= g["raw_conf"]:
                    print(f"🚫 ตัดกรอบแพะปลอมบนตัวคนออก (ซ้อนทับคน {round(overlap * 100)}%)")
                    is_false_on_human = True
                    break

            if not is_false_on_human:
                detections.append({
                    "box": g["box"],
                    "confidence": g["confidence"],
                    "class_id": g["class_id"],
                    "class_name": g["class_name"]
                })

        count = len(detections)
        print(f"📸 สรุปผล -> 🐐 AI นับแพะได้: {count} ตัว")

        del results, image, contents
        gc.collect()

        return {
            "success": True,
            "count": count,
            "detections": detections,
            "message": f"Found {count} goats"
        }

    except Exception as e:
        gc.collect()
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return {"success": False, "count": 0, "error": str(e)}

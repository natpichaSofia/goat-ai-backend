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
    dummy_img = Image.new("RGB", (320, 320))
    with torch.no_grad():
        model.predict(dummy_img, imgsz=320, verbose=False)
    del dummy_img
    gc.collect()
    print("✅ โหลดและวอร์มโมเดลแพะ (best.pt) พร้อมใช้งาน!")
except Exception as e:
    print(f"⚠️ โหลด best.pt ไม่สำเร็จ ({e}) -> ใช้ yolov8n.pt แทน")
    model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"status": "AI Server is online and ready!"}

@app.post("/count-goat")
async def count_goat(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        with torch.no_grad():
            results = model.predict(image, conf=0.15, iou=0.45, imgsz=320, verbose=False)

        detections = []
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            print(f"🔍 เจอวัตถุ: {class_name} (มั่นใจ {round(confidence * 100, 1)}%)")

            if class_name.lower() != "goat":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "box": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(confidence, 2),
                "class_id": class_id,
                "class_name": class_name
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

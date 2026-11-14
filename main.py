from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# ---------------------------------------------------------
# 1. ตั้งค่า CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 2. โหลดสมอง AI
# ---------------------------------------------------------
try:
    model = YOLO("best.pt")
    print("✅ โหลดโมเดลแพะ (best.pt) เรียบร้อย!")
    print("📋 คลาสที่โมเดลรู้จัก:", model.names)
except Exception as e:
    print(f"⚠️ ไม่เจอไฟล์ best.pt ({e}) -> กำลังใช้โมเดลมาตรฐาน yolov8n.pt แทน")
    model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"status": "AI Server is online and ready!"}

@app.post("/count-goat")
async def count_goat(file: UploadFile = File(...)):
    try:
        # 1. อ่านรูปภาพ 
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. ให้ AI ทำนาย (ปรับ conf ลดลงเหลือ 0.35 เพื่อให้จับภาพแพะบนหน้าจอได้ง่ายขึ้น)
      results = model.predict(image, conf=0.15, iou=0.45, imgsz=320)
        
        detections = []
        result = results[0]
        
        # 3. ดึงรายละเอียดของแต่ละกรอบ และกรองเอาเฉพาะคลาส "goat" (ไม่สนตัวพิมพ์เล็ก/ใหญ่)
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()
            
            # แสดงผลในจอดำทุกวัตถุที่เจอ จะได้รู้ว่า AI เห็นอะไรบ้างและมั่นใจกี่ %
            print(f"🔍 เจอวัตถุ: {class_name} (มั่นใจ {round(confidence * 100, 1)}%)")
            
            # กรองเอาเฉพาะคลาส goat (รองรับทั้ง "Goat" และ "goat")
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
        
        # 4. แสดงผลสรุปในจอดำ (Terminal)
        print(f"📸 สรุปผล -> 🐐 AI นับแพะได้: {count} ตัว")
        
        return {
            "success": True,
            "count": count,
            "detections": detections, 
            "message": f"Found {count} goats"
        }
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return {"success": False, "count": 0, "error": str(e)}

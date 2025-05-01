import numpy as np
import onnxruntime as ort
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse

# and your inference logic

app = FastAPI()


MODEL_PATH   = "model.onnx"
IMG_SIZE     = (150, 150)
CLASS_LABELS = ["Healthy", "Unhealthy", "Unknown"]

# ─── LOAD MODEL ───────────────────────────────────────────────
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ ONNX model loaded:", MODEL_PATH)
except Exception as e:
    print("❌ Failed to load ONNX model:", e)
    session = None

# ─── HELPER FUNCTIONS ─────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(image: Image.Image):
    batch = preprocess_image(image)
    results = session.run([output_name], {input_name: batch})
    probs = results[0][0]
    idx = int(np.argmax(probs))
    label = CLASS_LABELS[idx]
    confidence = float(probs[idx])
    return {
        "prediction": label,
        "confidence": confidence,
        "all_probs": {CLASS_LABELS[i]: float(probs[i]) for i in range(len(probs))}
    }

# ─── FASTAPI ENDPOINT ────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = predict_image(image)
    print("📦 Prediction:", result)
    return JSONResponse(content=result)

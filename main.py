from fastapi import FastAPI, UploadFile, File, Request
from app.schemas import PredictionResponse
from app.model_handler import predict_image
from PIL import Image
import io
import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from uuid import uuid4
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Web UI root --> for uploading images.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# JSON prediction API --> for Postman.
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = predict_image(image)
    return {"predictions": results}

# Web UI prediction API --> for web interface.
@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = predict_image(image)

    # Save uploaded image temporarily
    filename = f"{uuid4().hex}.png"
    img_path = f"app/static/uploads/{filename}"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    image.save(img_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "predictions": results,               # list of top-3
        "top_prediction": results[0],         # for easy access
        "image_url": f"/static/uploads/{filename}"
    })

# Check the life of the endpoints API
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Image Classifier"
    }
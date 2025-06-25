from fastapi import FastAPI, File, Request, UploadFile
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import shutil
import os

MODEL = tf.keras.models.load_model("/Users/harshnahata/Main/Deep Learning/cnn/agriculture-tech/saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Mount the uploads folder to serve files
app = FastAPI()
app.mount("/static", StaticFiles(directory="/Users/harshnahata/Main/Deep Learning/cnn/agriculture-tech/static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def read_file_as_image(data):
    try:
        image = Image.open(BytesIO(data))
        return np.array(image)
    except Exception as e:
        print(f"Error reading image: {str(e)}")
        raise



@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    upload_folder = "/Users/harshnahata/Main/Deep Learning/cnn/agriculture-tech/static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image for prediction
    try:
        file.file.seek(0)  # Reset pointer
        image = Image.open(file.file)  # Read the file as an image
        image = np.array(image)  # Convert to numpy array for further processing

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error reading image: {str(e)}"
            }
        )

    # Resize and preprocess the image
    img_batch = np.expand_dims(image, 0)

    # Make prediction
    folder_name = file.filename.split('_')
    foldername = folder_name[0]+"___"+folder_name[3]+"_"+folder_name[4]
    image_url = f"/Users/harshnahata/Main/Deep Learning/cnn/agriculture-tech/static/uploads/{file.filename}"
    print(image_url)

    predictions = MODEL.predict(img_batch)
    actual_class = foldername
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Render result
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "actual_class": actual_class, 
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "image_url": image_url  # Pass the uploaded image URL
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
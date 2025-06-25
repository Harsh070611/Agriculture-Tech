import os
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Assuming CLASS_NAMES and MODEL are already defined
MODEL = tf.keras.models.load_model("saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def read_file_as_image(data):
    # Function to read image data
    # Replace this with your image processing logic
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(image, (224, 224))

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        # Save the uploaded file to the server
        image_data = await file.read()
        folder_name = "static"  # Folder to save images
        os.makedirs(folder_name, exist_ok=True)  # Ensure the folder exists
        file_path = os.path.join(folder_name, file.filename)

        # Save image file
        with open(file_path, "wb") as f:
            f.write(image_data)

        # Read and process the image for prediction
        image = read_file_as_image(image_data)
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Error reading image: {str(e)}"}
        )

    # Preprocess the image
    img_batch = np.expand_dims(image, 0)

    # Predict class
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Render result with image URL
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "image_url": f"/{file_path}"  # Image URL for display
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
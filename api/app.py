from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
MODEL = tf.keras.models.load_model("/Users/harshnahata/Main/Deep Learning/cnn/agriculture-tech/saved_models/1.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # Serves the initial HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file:
            # Save the file securely
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Dummy prediction logic (replace with your ML model)
            predictions = MODEL.predict(img_batch)
            actual_class = foldername
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            return render_template(
                'index.html',
                image_url=url_for('static', filename=f'uploads/{filename}'),
                actual_class=actual_class,
                predicted_class=predicted_class,
                confidence=confidence
            )
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

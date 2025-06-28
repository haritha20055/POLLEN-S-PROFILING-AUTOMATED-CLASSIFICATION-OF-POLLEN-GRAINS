from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from reportlab.pdfgen import canvas
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('cnn_best.h5')

# Build label encoder from dataset classes
path = "data/"
classes = sorted(set(name.replace(' ', '_').split('_')[0] for name in os.listdir(path)))
le = LabelEncoder()
le.fit(classes)

# Track last prediction
last_prediction = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_prediction
    prediction = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"
        file = request.files['file']
        if file.filename == '':
            return "No selected file!"

        # Always save to static folder for preview
        filepath = os.path.join('static', 'last_uploaded.jpg')
        os.makedirs('static', exist_ok=True)
        file.save(filepath)

        # Load & preprocess
        img = load_img(filepath, target_size=(128,128))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
        y_pred = model.predict(x)
        class_idx = np.argmax(y_pred, axis=1)[0]
        class_name = le.inverse_transform([class_idx])[0]

        prediction = f"This looks like: {class_name}"
        last_prediction = f"Predicted class: {class_name}"

    return render_template("index.html", prediction=prediction)

@app.route('/download', methods=['GET'])
def download_report():
    global last_prediction
    file_path = "static/prediction_report.pdf"
    c = canvas.Canvas(file_path)
    c.setFont("Helvetica", 14)
    c.drawString(100, 750, "🌿 Pollen's Profiling - Classification Report")
    c.drawString(100, 700, last_prediction)
    c.save()
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)


model_path = "plant_disease_model.h5"

if os.path.exists(model_path):
    print("Model exists, loading...")
else:
    print("Model path not found!")

# Then try loading model
model = tf.keras.models.load_model(model_path)

# Paths
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES_PATH = "class_names.txt"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError("❌ class_names.txt not found. Train the model first.")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected.")

        # Save file
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Preprocess
        img = Image.open(filepath).convert("RGB").resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        prediction = f"{predicted_class} ({confidence:.2f}% confidence)"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

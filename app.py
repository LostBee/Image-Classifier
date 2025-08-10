# app.py  
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile, os
from tensorflow.keras.applications.resnet50 import ResNet50
from new_image_classifier_ResNet50_ import classify_image   # your function!

app = Flask(__name__)
CORS(app)                          # enable CORS for all routes

print("Loading ResNet50")
MODEL = ResNet50(weights="imagenet")
print("Model ready!")

# ----------- NEW ENDPOINT -------------
@app.route("/api/classify", methods=["POST"])
def classify_upload():
    # 1. sanity check
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400

    up_file = request.files["image"]
    if up_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # 2. save to a secure temp path
    filename = secure_filename(up_file.filename)        # strips evil ".."
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, filename)
        up_file.save(img_path)

        # 3. run the classifier
        preds = classify_image(img_path, MODEL)         # <-- reuse global model

    # 4. send JSON back
    return jsonify({"predictions": preds})

# --------------------------------------
@app.route("/")
def index():
    return "POST an image to /api/classify"

if __name__ == "__main__":
    app.run(debug=True)

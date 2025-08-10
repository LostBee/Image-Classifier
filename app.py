from flask import Flask
from tensorflow.keras.applications.resnet50 import ResNet50

app = Flask(__name__)

print("Loading ResNet50 model")
MODEL = ResNet50(weights="imagenet")     # happens ONE time, right now
print("Model ready!")

@app.route("/")
def hello():
    return "Model loaded -server is alive!"

if __name__ == "__main__":
    app.run(debug=True)

# image_classifier_ResNet50_.py
# Preparing to work with web server file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

def classify_image(image_path: str, model: ResNet50 = None, top_k: int = 3):
    """
    Classify an image and return a list of (label, confidence) tuples.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model : ResNet50, optional
        Pre-loaded model. If None, a new one is created (slower).
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    list[tuple[str, float]]
        e.g. [('tabby cat', 0.87), ('tiger cat', 0.08), ('Egyptian cat', 0.04)]
    """
    if model is None:
        model = ResNet50(weights="imagenet")   # one-off load

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top_k)[0]

    # keep only label and probability
    return [(label.replace("_", " "), float(prob)) for (_, label, prob) in decoded]


# Stand-alone test: `python new_image_classifier_ResNet50_.py example.jpg`
if __name__ == "__main__":
    import sys, pprint, os

    if len(sys.argv) < 2:
        print("Usage: python new_image_classifier_ResNet50_.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    print("Loading model...")
    model = ResNet50(weights="imagenet")      # load once
    results = classify_image(img_path, model)
    pprint.pprint(results)

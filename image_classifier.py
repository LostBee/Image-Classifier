#Importing ingredients as needed
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

#The recipe - fucntion to classify images
def classify_image(image_path):
    # ... all the steps are inside here ...
    
# Let's bring the master chef before we start cooking - The AI Model itself
# Loading pre-trained model MobileNetV2
# This model was trained on the ImageNet dataset, which has 1000 categories
 model = MobileNetV2(weights='imagenet')
 
# Load and preprocess the image
# The model expects images to be exactly 224x224 pixels.
img = image.load_img(image_path, target_size=(224, 224))

# Convert the image to a format the model understands (a numpy array)
img_array = image.img_to_array(img)

# Add an extra dimension because the model expects a "batch" of images
img_array_expanded = np.expand_dims(img_array, axis=0)

# Normalize the image data to match the format the model was trained on
processed_img = preprocess_input(img_array_expanded)

 
 

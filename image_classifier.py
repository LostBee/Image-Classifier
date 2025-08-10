#Importing ingredients as needed
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

#The recipe - fucntion to classify images
def classify_image(image_path):
    # ... all the steps are inside here ...
    

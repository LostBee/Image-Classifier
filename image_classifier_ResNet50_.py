#Importing ingredients as needed
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

#The recipe - fucntion to classify images
def classify_image(image_path):
        
    # Let's bring the master chef before we start cooking - The AI Model itself
    # Loading pre-trained model MobileNetV2
    # This model was trained on the ImageNet dataset, which has 1000 categories
    
    model = ResNet50(weights='imagenet')

    # Load and preprocess the image (ResNet50 also uses 224x224)
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Use the new preprocess_input function imported from resnet50
    processed_img = preprocess_input(img_array_expanded)

    # Make a prediction
    predictions = model.predict(processed_img)

    # Decode the results into human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions
 
#Main script
if __name__ == "__main__":
    # IMPORTANT: Change this to the path of your image file
    IMAGE_FILE_PATH = 'image2.jpg' 

    try:
        results = classify_image(IMAGE_FILE_PATH)
        print("The model predicts this is a...")
        for i, (imagenet_id, label, score) in enumerate(results):
            print(f"{i+1}: {label.replace('_', ' ')} ({score:.2%})")
    except FileNotFoundError:
        print(f"Error: The file '{IMAGE_FILE_PATH}' was not found.")
        print("Please make sure the image is in the same folder as the script, or provide the full path.")
 

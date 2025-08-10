# Simple Image Classifier with TensorFlow

This project contains multiple Python script that uses a pre-trained deep learning model to classify the contents of an image. It can load powerful models like `MobileNetV2` and `ResNet50` to identify objects from 1000 different categories.

This was built as part of a guided learning session to be hands on and understad the fundamentals of using pre-trained models in machine learning.

## Features

* Classifies local image files (`.jpg`, `.png`, etc.).
* Uses powerful, pre-trained models from TensorFlow/Keras.
* Easily switch between different models (e.g., `MobileNetV2`, `ResNet50`).
* Prints the top 3 predictions with their confidence scores.

## Technology Stack

* **Python**
* **TensorFlow / Keras:** For loading models and making predictions.
* **Pillow:** For image manipulation.
* **NumPy:** for numerical operations.

## Setup and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/LostBee/Image-Classifier.git](https://github.com/LostBee/Image-Classifier.git)
    cd Image-Classifier
    ```

2.  **Install the required libraries:**
    ```bash
    pip install tensorflow Pillow
    ```

3.  **Run the script:**
    * Place an image you want to classify inside the project folder.
    * Open the `image_classifier.py` file and change the `IMAGE_FILE_PATH` variable to your image's name.
    * Execute the script from your terminal:
        ```bash

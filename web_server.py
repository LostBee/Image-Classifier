#Setup
# web_server.py

from flask import Flask, request, jsonify
from flask_cors import CORS

# Create the app and enable CORS
app = Flask(__name__)
CORS(app)

# Define the URL endpoint that will accept our image
@app.route('/api/classify', methods=['POST'])
def classify_upload():
    # Check if an image file was sent with the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    image_file = request.files['image']
    
    # Check if the user submitted the form without selecting a file
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # For now, we'll just confirm we received it.
    # The actual AI classification logic will go here later.
    print(f"Received file: {image_file.filename}")
    
    # Send a success message back
    return jsonify({'message': f'Successfully received {image_file.filename}'})

# This runs the server when you execute the script
if __name__ == '__main__':
    app.run(debug=True)
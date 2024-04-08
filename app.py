
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image as img_preprocessing
import numpy as np
from keras.applications.vgg16 import preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS 
import os  # Import the os module

app = Flask(__name__)
CORS(app)  

# Load the VGG16 model
model = keras.models.load_model("models\\EfficientNetB0_model.h5")

# Define function for prediction
def predict(image_path, model):
    img = img_preprocessing.load_img(image_path, target_size=(224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    classes = ['Age Degeneration', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Normal', 'Others']
    return classes[np.argmax(predictions)]

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_eye_condition():
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'})

    # Get the image file from the request
    image_file = request.files['image']

    # Save the image file temporarily
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)

    # Make prediction using the model
    condition = predict(temp_image_path, model)

    # Delete the temporary image file
    os.remove(temp_image_path)

    # Return the prediction result
    return jsonify({'predicted_condition': condition})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

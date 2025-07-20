from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model once when the app starts
model = load_model("D:/internship/animalClassifier1.h5")

# Route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file temporarily
    filepath = os.path.join('./', file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Make prediction
    prediction = np.argmax(model.predict(x))
    classes = ['bears', 'crows', 'elephants', 'rats']
    result = classes[prediction]

    # Optionally remove the temporary file after prediction
    os.remove(filepath)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

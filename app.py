from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()


# Load your trained model
         # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
#model_best = load_model('papaya.hdf5',compile = False)

print('Model loaded. Check http://127.0.0.1:5000/')

def predict_class(model, images, show = True):
  food_list = ['anthracnose','black_spot','phytophthora','powdery_mildew','ring_spot']
  for img in images:
    img = image.load_img(img, target_size=(32,32))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.
    print("In lopp of prdict class")                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    pred_value = food_list[index]
    print("in predict class")
    return pred_value
    



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        food_list = ['anthracnose','black_spot','phytophthora','powdery_mildew','ring_spot']
        model_best = load_model('papaya.hdf5',compile = False)
        images = []
        images.append(file_path)
        # Make prediction
        preds = predict_class(model_best , images)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
                      # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)


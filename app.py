from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os

from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo
# from pymongo import MongoClient

from pymongo.errors import DuplicateKeyError, OperationFailure
from bson.objectid import ObjectId
from bson.errors import InvalidId

import tensorflow as tf
import keras
import cv2
import numpy as np

# from flask_mongoengine import MongoEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = "desertBasser"
app.config["MONGO_URI"] = "mongodb://localhost:27017/Desert"
mongo = PyMongo(app)

UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model_path = 'models/baseer.h5'
model = tf.keras.models.load_model(model_path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):

            lat = float(request.form['latitude'])
            lng = float(request.form['longitude'])

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            predictions = predict("static/uploads/" + filename)
            postprocessing = postprocess(predictions)

            save_to_database(filename, lat, lng, postprocessing)

            return render_template('index.html', filename=filename)
        else:
            flash('Allowed image types are - png, jpg')
            return redirect(request.url)


def save_to_database(filename, lat, lng, output):
    mongo.db.reports.insert_one({
        'img': filename,
        'result': output,
        'lat': lat,
        'long': lng,
    })


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def preprocess_image(img):
    img = cv2.imread(img)
    processed = cv2.resize(img, (224, 224))
    img_arry = keras.utils.img_to_array(processed)
    return img_arry


def predict(img):
    processed = preprocess_image(img)

    predictions = model.predict(np.expand_dims(processed, axis=0))
    return predictions


def postprocess(predictions):
    class1_prob = predictions[0]

    result = f"The Image is {100 * class1_prob}% Street and {100 * (1 - class1_prob)}% Sand."

    if 100 * (1 - class1_prob) > 50:
        flash(
            f"The Image is {100 * class1_prob}% Street and {100 * (1 - class1_prob)}% Sand.")
        flash('Warning!!')
    else:
        flash('Save Road')

    return result


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='static/desert.icon')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.debug = True
    app.run()


""""""""""
Author: Binary mask & semantic segmentation 
Copyright (C) 2023 
Mujtab Almeshal
Hassan Alhathiq
Saad Almunif

Version: 1.0v


"""""""""

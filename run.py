from flask import Flask, jsonify, make_response, request, render_template, redirect, url_for
import os
from utils.functions import allow_image, process_img
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "./static/img/uploads/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def jinja():
    # Strings
    my_name = "Julian"

    # Integers
    my_age = 30

    # Lists
    langs = ["Python", "JavaScript", "Bash", "Ruby", "C", "Rust"]

    # Dictionaries
    friends = {
        "Tony": 43,
        "Cody": 28,
        "Amy": 26,
        "Clarissa": 23,
        "Wendell": 39
    }

    # Tuples
    colors = ("Red", "Blue")

    # Booleans
    cool = True

    # Classes
    class GitRemote:
        def __init__(self, name, description, domain):
            self.name = name
            self.description = description 
            self.domain = domain

        def clone(self, repo=''):
            return f"Cloning into {repo}"

    my_html = '<h1>This is some HTML</h1>'

    my_remote = GitRemote(
        name="Learning Flask",
        description="Learn the Flask web framework for Python",
        domain="https://github.com/Julian-Nash/learning-flask.git"
    )

    # Functions
    def repeat(x, qty=1):
        return x * qty

    return render_template(
        "about.html", my_name=my_name, my_age=my_age, langs=langs,
        friends=friends, colors=colors, cool=cool, GitRemote=GitRemote, 
        my_remote=my_remote, repeat=repeat, my_html=my_html
    )

@app.route("/result", methods = ['POST', 'GET'])
def sign_up():
    if request.method == 'POST':
        req = request.form
        
        missing = list()

        for k, v in req.items():
            if v == '':
                missing.append(k)
        
        if missing:
            feedback = f"Missing fields for {', '.join(missing)}"
            return render_template("result.html", feedback=feedback)

        return redirect(request.url)
    return render_template("result.html")

@app.route('/diagnosis', methods = ['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']

            if allow_image(image.filename, app.config["ALLOWED_IMAGE_EXTENSIONS"]):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                print("{} saved".format(filename))

                new_model = keras.models.load_model('model.h5')
                img_path = './static/img/uploads/' + os.listdir('./static/img/uploads/')[0]
                test_img = process_img(img_path, 150, 32)
                prediction = new_model.predict(test_img)[0][0]
                print("CLASS >>>> {}".format(str(prediction)))

                return render_template('diagnosis.html', label = filename ,filename = app.config["IMAGE_UPLOADS"] + filename)
            else:
                feedback = "File format not allowed."
                return render_template("diagnosis.html", feedback=feedback)
            
            return redirect(request.url)
        
    return render_template('diagnosis.html')

if __name__ == '__main__':
    app.run()
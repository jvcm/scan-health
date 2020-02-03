from flask import Flask, jsonify, make_response, request, render_template, redirect, url_for
import os
from utils.functions import allow_image, process_img
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import keras
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

p = [0,0]

app = Flask(__name__)
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

@app.route('/diagnosis', methods = ['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if request.files:
            image = request.files['imgInp']

            if allow_image(image.filename, app.config["ALLOWED_IMAGE_EXTENSIONS"]):

                #Load CNN model
                new_model = keras.models.load_model('./model/model.h5')

                #Image preprocessing and classification
                test_img = process_img(image, 150, 32)
                prediction = new_model.predict(test_img)[0][0]
                prob = 100*round(prediction, 3)
                classes = ['Normal', 'Pneumonia']
                pred_class = classes[int(round(prediction))]

                global p 
                p = [100 - prob, prob]

                return render_template('result.html', classification = pred_class, probability = prob)
            else:
                feedback = "File format not allowed."
                return render_template("diagnosis.html", feedback=feedback)
                    
    return render_template('diagnosis.html')

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():

    fontsize= 12
    plt.rcParams['font.size'] = fontsize
    cols = ['#66b3ff', '#ff9999']

    fig = plt.Figure(figsize = (3.5,3.5))
    axis = fig.add_subplot(1, 1, 1)
    axis.pie(p, autopct='%.1f%%', colors= cols)
    fig.legend(labels=['Normal', 'Pneumonia'])

    circle = plt.Circle(xy=(0, 0), radius=0.7, facecolor='white')
    fig.gca().add_artist(circle)

    return fig


if __name__ == '__main__':
    app.run()
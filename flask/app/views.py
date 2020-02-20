from app import app
from flask import request, render_template, Markup
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import gc
from app import sess, graph, new_model
from tensorflow.python.keras.backend import set_session


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/diagnosis', methods = ['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if request.files:
            image = request.files['imgInp']

            if allow_image(image.filename, ["JPEG", "JPG", "PNG"]):

                # Image preprocessing and classification.
                global sess
                global graph
                global new_model
                test_img = process_img(image, 150, 32)
                with graph.as_default():
                    set_session(sess)
                    prediction = new_model.predict(test_img, verbose = 1)[0][0]

                prob = 100*round(prediction, 3)
                classes = ['Normal', 'Pneumonia']
                pred_class = classes[int(round(prediction))]

                c = [100 - prob, prob]

                imagem = get_encoded_image(c)

                # Memory release
                gc.collect()

                # The result.html page will be rendered with 3 objects:
                #   - The class of uploaded image;
                #   - Probability of 'pneumonia' class;
                #   - Generated donut plot containing the class probabilities within the HTML code.
                return render_template('result.html',
                    classification = pred_class, 
                    probability = prob, 
                    img_code= Markup('<img src=data:image/png;base64,{} style="height:250px; width:auto;" alt="Responsive image">'.format(imagem)))
            else:
                # In case file is no choosen or file format is invalid.
                feedback = "File format not allowed."
                return render_template("diagnosis.html", feedback=feedback)
                    
    return render_template('diagnosis.html')

def allow_image(filename, imageExtensions):
    # Check if image file is valid before uploading it.
    if filename == "":
        print("No filename")
        return False
    elif not "." in filename:
        print("No . in filename")
        return False
    elif not filename.split('.')[-1].upper() in imageExtensions:
        print("Invalid format")
        return False
    else:
        return True

def process_img(img_path, img_dims, batch_size):
    # This method is responsible for receiving an image of any given size and adapt to the input
    # of our CNN model. The image is converted to gray scale and resized to 150x150 px.
    test_data = []

    img = plt.imread(img_path)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_dims, img_dims))
    img = np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
        
    test_data = np.array(test_data)
    
    return test_data

def get_encoded_image(c):
    # This method is responsible for generating an image and not saving as a static file in the server.
    # Instead, BytesIO can be used like a file opened in binary mode.
    fontsize= 12
    plt.rcParams['font.size'] = fontsize
    cols = ['#66b3ff', '#ff9999']

    fig = plt.Figure(figsize = (3.5,3.5))
    axis = fig.add_subplot(1, 1, 1)
    axis.pie(c, autopct='%.1f%%', colors= cols)
    fig.legend(labels=['Normal', 'Pneumonia'])

    # The donut plot is generated with matplotlib and will be rendered within the result.html file.
    circle = plt.Circle(xy=(0, 0), radius=0.7, facecolor='white')
    fig.gca().add_artist(circle)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url
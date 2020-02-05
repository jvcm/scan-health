from flask import Flask, request, render_template, Markup
import os
from utils.functions import allow_image, process_img
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64

app = Flask(__name__)
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template("about.html")

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

                c = [100 - prob, prob]

                imagem = get_encoded_image(c)

                return render_template('result.html',
                    classification = pred_class, 
                    probability = prob, 
                    img_code= Markup('<img src=data:image/png;base64,{} style="height:250px; width:auto;" alt="Responsive image">'.format(imagem)))
            else:
                feedback = "File format not allowed."
                return render_template("diagnosis.html", feedback=feedback)
                    
    return render_template('diagnosis.html')

def get_encoded_image(c):

    fontsize= 12
    plt.rcParams['font.size'] = fontsize
    cols = ['#66b3ff', '#ff9999']

    fig = plt.Figure(figsize = (3.5,3.5))
    axis = fig.add_subplot(1, 1, 1)
    axis.pie(c, autopct='%.1f%%', colors= cols)
    fig.legend(labels=['Normal', 'Pneumonia'])

    circle = plt.Circle(xy=(0, 0), radius=0.7, facecolor='white')
    fig.gca().add_artist(circle)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

if __name__ == '__main__':
    app.run(host="0.0.0.0")
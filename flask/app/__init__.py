from flask import Flask
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

def load():
    # This is a way to load the CNN model once in the application,
    # avoiding the increase of memory usage.
    global new_model, graph, sess
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    # Convolutional Neural Network model file is located in this folder
    new_model = load_model('app/model.h5')

# After creating the callable 'app' object, the model is loaded using a global variable, accessible 
# throughout all views in the application. 
app = Flask(__name__, static_folder="./static/")
load()

from app import views
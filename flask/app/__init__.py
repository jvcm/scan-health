from flask import Flask
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

def load():
    global new_model, graph, sess
    sess = tf.Session()
    graph = tf.get_default_graph()
    set_session(sess)
    new_model = load_model('app/model.h5')

app = Flask(__name__)
load()

from app import views
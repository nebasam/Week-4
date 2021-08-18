from json import load
from flask import Flask, render_template, send_from_directory, request
import os
from librosa.feature.spectral import mfcc   
import pandas as pd
import pickle
import sys
import os
import shutil
import numpy as np
import librosa
# import tensorflow as tf
from tensorflow.keras.models import load_model
# import pyaudio
from deepspeech import Model
import scipy.io.wavfile as wav
import wave
from werkzeug.utils import secure_filename, send_file
from tensorflow.keras import backend as K
from scripts.utils import  int_sequence_to_text
from scripts.Models import model_2


app = Flask(__name__)


MODEL_PATH = './model/deep.h5'

@app.route("/assets/<path:path>")
def static_dir(path):
    return send_from_directory("static/assets", path)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")



def make_prediction():

    deep_speech = model_2(input_dim=26,
                filters=100,
                kernel_size=4, 
                conv_stride=1,
                conv_border_mode='valid',
                units=250,
                activation='tanh',
                dropout_rate=0.2,
                number_of_layers=2,
                output_dim=29)

    data,sample_rate=librosa.load('./uploads/audio/a.wav')
    mfccs = librosa.feature.mfcc(data, sr=16000,n_mfcc=26)
 
    # def make_predictions(model,features):
    predictions=[]
    deep_speech.load_weights(MODEL_PATH)
    # model = load_weight(MODEL_PATH)
    for i in [mfccs]:
        data_point=i.T
        prediction = deep_speech.predict(np.expand_dims(i.T, axis=0))
        output_length = [deep_speech.output_length(data_point.shape[0])]
        pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length, greedy=False)[0][0])+1).flatten().tolist()
        predicted = ''.join(int_sequence_to_text(pred_ints)).replace("<SPACE>", " ")

        predictions.append(predicted)
    return predictions

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global class_names
    if request.method == 'POST':
       
        # Get the file from post request
        
        f = request.files['audio']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/audio', secure_filename(f.filename))
        f.save(file_path)
        new_file = os.path.join(os.path.join(os.path.join(
            basepath, 'uploads/audio')), "a.wav")
       
        os.rename(file_path, new_file)
        # print(f)

        #Make Prediction
        preds = make_prediction()
        shutil.rmtree('./uploads/audio')
        os.mkdir('./uploads/audio')


        return preds[0]


@app.route("/about")
def about():
    return "<h1>About</h1>"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host="0.0.0", debug=True,port=port)
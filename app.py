from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np


app = Flask(__name__)
model = tf.keras.models.load_model('model')
tokenizer = Tokenizer()

@app.route('/',methods=['GET','POST'])
def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def make_lyrics(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return render_template("index.html",out_data=song)
song=make_lyrics("Parayo",2)

if __name__ == "__main__":
    app.run(debug=True)
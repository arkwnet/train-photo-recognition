import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify
 
app = Flask(__name__)
 
@app.route("/")
def main():
    input_filename = sys.argv[2]
    classes = ["keikyu-1000", "meitetsu-3500", "meitetsu-6000"]
    num_classes = len(classes)
    img_width, img_height = 128, 128
    feature_dim = (img_width, img_height, 3)
    model = tf.keras.models.load_model(sys.argv[1])
    img = image.load_img(input_filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    pred = model.predict(x)[0]
    result = {}
    for cls, prob in zip(classes, pred):
        result[cls] = str(prob)
    return jsonify(result)
 
if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 8888, debug = True)
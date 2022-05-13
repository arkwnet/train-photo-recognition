import sys
import io
import datetime
import random
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
 
app = Flask(__name__)
model = tf.keras.models.load_model(sys.argv[1])
 
@app.route("/", methods=["GET","POST"])
def main():
    if request.method == "POST":
        time_delta = datetime.timedelta(hours = 9)
        time_jst = datetime.timezone(time_delta, "JST")
        time = datetime.datetime.now(time_jst)
        time_d = time.strftime("%Y%m%d%H%M%S")
        image_file = "./static/upload/" + time_d + "_" + str(random.randint(1000000,9999990)) + ".jpg"
        with open(image_file, "bw") as f:
            post_data = request.get_json()
            img_base64 = post_data["image"]
            img_binary = base64.b64decode(img_base64.replace("data:image/jpeg;base64,", "").encode())
            f.write(img_binary)
        classes = ["keikyu-1000", "meitetsu-3500", "meitetsu-6000"]
        num_classes = len(classes)
        img_width, img_height = 128, 128
        feature_dim = (img_width, img_height, 3)
        img = image.load_img(image_file, target_size=(img_height, img_width))
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
# 参考 : https://qiita.com/everylittle/items/954207b1ae917c25ff96#%E3%83%87%E3%83%BC%E3%82%BF%E9%9B%86%E3%82%81

import sys

# 入力画像
input_filename = "sample-3500-1.jpg"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 特徴量の設定
classes = ["keikyu-1000", "meitetsu-3500", "meitetsu-6000"]
num_classes = len(classes)
img_width, img_height = 128, 128
feature_dim = (img_width, img_height, 3)

# 学習モデルの読み込み
model = tf.keras.models.load_model("train11.hdf5")

# 入力画像の読み込み
img = image.load_img(input_filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 学習時と同様に値域を[0, 1]に変換する
x = x / 255.0

# 車両形式を予測
pred = model.predict(x)[0]

# 結果を表示する
for cls, prob in zip(classes, pred):
    print("{0:18}{1:8.4f}%".format(cls, prob * 100.0))
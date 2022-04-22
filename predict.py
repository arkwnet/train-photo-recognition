import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 入力画像
input_filename = sys.argv[2]

# 特徴量の設定
classes = ["keikyu-1000", "meitetsu-3500", "meitetsu-6000"]
num_classes = len(classes)
img_width, img_height = 128, 128
feature_dim = (img_width, img_height, 3)

# 学習モデルの読み込み
model = tf.keras.models.load_model(sys.argv[1])

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
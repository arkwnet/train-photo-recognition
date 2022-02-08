# 参考 : https://qiita.com/everylittle/items/954207b1ae917c25ff96#%E3%83%87%E3%83%BC%E3%82%BF%E9%9B%86%E3%82%81

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 学習設定
batch_size = 32
epochs = 30
# 特徴量の設定
# classesはサブフォルダの名前に合わせる
classes = ["keikyu-1000", "meitetsu-3500", "meitetsu-6000"]
num_classes = len(classes)
img_width, img_height = 128, 128
feature_dim = (img_width, img_height, 3)
# ファイルパス
data_dir = "./images"

# 画像の準備
datagen = ImageDataGenerator(
    rescale=1.0 / 255, # 各画素値は[0, 1]に変換して扱う
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)
train_generator = datagen.flow_from_directory(
    data_dir,
    subset="training",
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True
)
validation_generator = datagen.flow_from_directory(
    data_dir,
    subset="validation",
    target_size=(img_width, img_height),
    color_mode="rgb",
    classes=classes,
    class_mode="categorical",
    batch_size=batch_size)

# 画像数を取得し、1エポックのミニバッチ数を計算
num_train_samples = train_generator.n
num_validation_samples = validation_generator.n
steps_per_epoch_train = (num_train_samples-1) // batch_size + 1
steps_per_epoch_validation  = (num_validation_samples-1) // batch_size + 1

# モデルの定義
# 学習済みのVGG16モデルをベースに、出力層だけを変えて学習させる
# block4_poolまでのパラメータは学習させない
vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=feature_dim)
for layer in vgg16.layers[:15]:
    layer.trainable = False

# モデルの構築
layer_input = Input(shape=feature_dim)
layer_vgg16 = vgg16(layer_input)
layer_flat = Flatten()(layer_vgg16)
layer_fc = Dense(256, activation="relu")(layer_flat)
layer_dropout = Dropout(0.5)(layer_fc)
layer_output = Dense(num_classes, activation="softmax")(layer_dropout)
model = Model(layer_input, layer_output)
model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(lr=1e-3, momentum=0.9),
              metrics=["accuracy"])

# 学習
cp_cb = ModelCheckpoint(
    filepath="train{epoch:02d}.hdf5",
    monitor="val_loss",
    verbose=1,
    mode="auto"
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    verbose=1
)
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    callbacks=[cp_cb, reduce_lr_cb]
)
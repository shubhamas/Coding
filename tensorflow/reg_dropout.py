import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)  = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)  
print(y_test.shape)

model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        layers.Conv2D(32, 3, padding = 'valid', activation = 'relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, padding = 'valid', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation = 'relu'),
        layers.Flatten(),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(10),
    ]
)

def my_model():
    inputs = keras.Input(shape=(28,28,3))
    x = layers.Conv2D(64, 3, stride=(1,1), padding='valid',kernel_regularizer=regularizers.l2(0.2),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, stride=(1,1), padding='valid',kernel_regularizer=regularizers.l2(0.2),)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.2),)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs = inputs, outputs=outputs)
    return model

model = my_model()
print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.SGD(lr = 3e-4),
    metrics = ['accuracy'],
)

model.fit(x_train, y_train, batch_size = 64, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=64,verbose=2)
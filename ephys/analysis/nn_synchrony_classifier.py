
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(100 * 100,), input_shape=(100, 100)),
    keras.layers.Dense(units=1000, activation='sigmoid'),
])
import tensorflow as tf
from tensorflow import keras
import numpy as np

class SumRows(keras.layers.Layer):
    def __init__(self):
        super().__init__(trainable=False)

    def call(self, inputs):
        row_sum = tf.reduce_sum(inputs, 1, keepdims=True)
        return tf.concat([inputs, row_sum], 1)


def build_model():
    inputs = keras.Input(shape=(1,))
    cat = keras.layers.Concatenate(axis=-1)
    x =  cat([inputs, inputs, inputs])
    outputs = SumRows()(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="dumb_model")


model = build_model()
model.save("model.h5")
with open("model_structure.json", "w") as json_file:
    json_file.write(model.to_json(indent=4))

input = np.linspace(1, 100, 10).reshape(10, 1)
with open("input.dat", "wb") as outfile:
    np.save(outfile, input)

result = model.predict(input, batch_size=10)
with open("result.dat", "wb") as outfile:
    np.save(outfile, result)


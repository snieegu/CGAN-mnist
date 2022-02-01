import keras
from keras import layers
import numpy as np

latent_dim = 28

generator_input = keras.Input(shape=latent_dim)

x = layers.Dense(128*28*28)(generator_input)
x = layers.leakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')
x = layers.LeakyReLU()(x)

x = layers.Conv2




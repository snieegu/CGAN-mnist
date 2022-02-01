from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras_preprocessing import image
import numpy as np
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices()
for dev in physical_devices:
    print(dev)

latent_dim = 28
epochs = 200
chanels = 1
height = 28
width = 28
learning_rate = 0.0002

generator_input = keras.Input(shape=latent_dim)

x = layers.Dense(128 * 28 * 28)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((28, 28, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, strides=1, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(chanels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()

discriminator_input = layers.Input(shape=(height, width, chanels))

x = layers.Conv2D(128, chanels)(discriminator_input)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, chanels, strides=2)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, chanels, strides=2)(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, chanels, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x - layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

(training_data, y_train), (_, _) = mnist.load_data()
print(training_data.shape)

training_data = training_data.reshape((training_data.shape[0],) + (height, width, chanels)).astype('float32') / 255

batch_size = 128

if not os.path.exists('generatedImg'):
    os.makedirs('generatedImg')
save_dir = 'generatedImg'

start = 0
for epoch in range(epochs):
    latent_vector = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(latent_vector)

    stop = start + batch_size
    real_images = training_data[start:stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

    labels += 0.05 * np.random.random(labels.shape)

    D_loss = discriminator.train_on_batch(combined_images, labels)

    latent_vector = np.random.normal(size=[batch_size, latent_dim])

    missleading_targets = np.zeros((batch_size, 1))

    G_loss = gan.train_on_batch(latent_vector, missleading_targets)

    start += batch_size
    if start > len(training_data) - batch_size:
        start = 0

    print("Working epoch:", epoch)

    if epoch % 50 == 0:
        gan.save_weights('gan.h5')

        print('Discriminator loss in epoch %s: %s' % (epoch, D_loss))
        print('oposite loss: %s %s' % (epoch, G_loss))

    if epoch % 10 == 0:
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_mnist' + str(epoch) + '.png'))

        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_mnist' + str(epoch) + '.png'))
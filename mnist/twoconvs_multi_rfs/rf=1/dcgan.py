# -*- coding: utf-8 -*-

"""
    test the generation of gan in which the discriminator of two layers is used.
"""

import glob
import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

tf.enable_eager_execution()

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 600000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


class Generator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(
            7 * 7 * 256, use_bias=False, input_shape=(100, ))
        self.bn1 = keras.layers.BatchNormalization()
        self.lrelu1 = keras.layers.LeakyReLU()

        self.reshape1 = keras.layers.Reshape((7, 7, 256))

        self.deconv2 = keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.lrelu2 = keras.layers.LeakyReLU()

        self.deconv3 = keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn3 = keras.layers.BatchNormalization()
        self.lrelu3 = keras.layers.LeakyReLU()

        self.deconv4 = keras.layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False, activation='tanh')

    @tf.function
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)

        x = self.reshape1(x)

        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)

        x = self.deconv3(x)
        x = self.bn3(x, training=training)
        x = self.lrelu3(x)

        x = self.deconv4(x)

        return x


class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = keras.layers.Conv2D(64, (1, 1), strides=(
            1, 1), padding='same', input_shape=[28, 28, 1])
        self.lrelu1 = keras.layers.LeakyReLU()

        self.conv2 = keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same')
        self.lrelu2 = keras.layers.LeakyReLU()

        self.flat3 = keras.layers.Flatten()
        self.dense3 = keras.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)

        x = self.flat3(x)
        x = self.dense3(x)

        return x


generator = Generator()
discriminator = Discriminator()
noise = tf.random_normal([1, 100])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, ..., 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)

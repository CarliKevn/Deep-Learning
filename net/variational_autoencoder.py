from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import pandas as pd

# Taken from Keras offical example of a variatonal encoder
class VariationalAutoencoder:
    def __init__(self, x_train, x_test, mse=False, weights=None, intermediate_dim=512, batch_size=128, latent_dim=2, epochs=200):
        self._x_train = x_train
        self._x_test = x_test
        self._mse = mse
        self._weights = weights
        self._original_dim = self._x_train.shape[1]
        # network parameters
        self._input_shape = (self._original_dim, )
        self._intermediate_dim = intermediate_dim
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._train()

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def _sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _buildModel(self):
        # VAE model = encoder + decoder
        # build encoder model
        self._inputs = Input(shape=self._input_shape, name='encoder_input')
        x = Dense(self._intermediate_dim, activation='relu')(self._inputs)
        self._z_mean = Dense(self._latent_dim, name='z_mean')(x)
        self._z_log_var = Dense(self._latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self._sampling, output_shape=(self._latent_dim,), name='z')([self._z_mean, self._z_log_var])

        # instantiate encoder model
        self._encoder = Model(self._inputs, [self._z_mean, self._z_log_var, z], name='encoder')
        self._encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(self._latent_dim,), name='z_sampling')
        x = Dense(self._intermediate_dim, activation='relu')(latent_inputs)
        self._outputs = Dense(self._original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self._decoder = Model(latent_inputs, self._outputs, name='decoder')
        self._decoder.summary()

        # instantiate VAE model
        self._outputs = self._decoder(self._encoder(self._inputs)[2])
        self._vae = Model(self._inputs, self._outputs, name='vae_mlp')


    def generate(self):
        z_mean, _, _ = self._encoder.predict(self._x_train, batch_size=self._batch_size)
        vae_data = self._decoder.predict(z_mean)
        vae_selected_feature = pd.DataFrame({'Fermeture': vae_data[:, 0], 'MACD': vae_data[:, 1], 'ema': vae_data[:, 2], 'wr': vae_data[:, 3]})
        return vae_selected_feature

    def _train(self):
        self._buildModel()
        models = (self._encoder, self._decoder)

        # VAE loss = mse_loss or xent_loss + kl_loss
        if self._mse:
            reconstruction_loss = mse(self._inputs, self._outputs)
        else:
            reconstruction_loss = binary_crossentropy(self._inputs,
                                                  self._outputs)

        reconstruction_loss *= self._original_dim
        kl_loss = 1 + self._z_log_var - K.square(self._z_mean) - K.exp(self._z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self._vae.add_loss(vae_loss)
        self._vae.compile(optimizer='adam')
        self._vae.summary()

        if self._weights:
            self._vae.load_weights(self._weights)
        else:
            # train the autoencoder
            self._vae.fit(self._x_train,
                    epochs=self._epochs,
                    batch_size=self._batch_size,
                    validation_data=(self._x_test, None),
                    shuffle=False)

            self._vae.save_weights('vae_features.h5')



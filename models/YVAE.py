import tensorflow as tf
from tensorflow.keras import layers

import sys

sys.path.append('models')
from modules import *
from losses import *


class Encoder(tf.keras.Model):
    def __init__(self,
                 c_dim=7,
                 z_dim=128,
                 dim=64,
                 max_filters=512,
                 norm='bn',
                 act='lrelu',
                 num_downsamples=6):
        super().__init__()

        self.blocks = tf.keras.Sequential([
            ConvBlock(dim, 7, 1, norm=norm, act=act)
        ])

        for i in range(1, num_downsamples):
            filters = min(dim * 2 ** i, max_filters)
            self.blocks.add(ResBlock(filters, norm, act, downsample=True))

        self.blocks.add(layers.Flatten())
        self.to_z = LinearBlock(2 * z_dim, act='none')
        self.to_c = LinearBlock(c_dim, act='none')

    def call(self, x, training=False):
        x = self.blocks(x, training=training)
        factors = self.to_z(x, training=training)
        factors = tf.split(factors, 2, axis=-1)
        logits = self.to_c(x)
        return factors, logits


class Decoder(tf.keras.Model):
    def __init__(self,
                 c_dim=7,
                 max_filters=512,
                 norm='bn',
                 act='lrelu',
                 num_downsamples=6,
                 num_resblocks=2
                 ):
        super().__init__()

        self.blocks = tf.keras.Sequential()

        for i in range(num_downsamples):
            if i<num_downsamples//2:
                filters = max_filters
            else:
                filters = max_filters // 2 ** (i-2)

            self.blocks.add(ConvBlock(filters, 5, 2, norm=norm, act=act, transpose=True))

        self.to_rgb = ConvBlock(3, 5, 2, act='sigmoid',transpose=True)

        self.c_dim = c_dim

    def call(self, inputs, training=False):
        c, z = inputs
        c = tf.one_hot(c, self.c_dim, axis=-1)
        x = tf.concat([c, z], axis=-1)
        x = tf.reshape(x, [x.shape[0], 1, 1,-1])
        x = self.blocks(x, training=training)
        x = self.to_rgb(x)
        return x


class Generator(tf.keras.Model):
    def __init__(self,
                 c_dim=7,
                 z_dim=128,
                 dim=64,
                 max_filters=512,
                 norm='bn',
                 act='lrelu',
                 num_downsamples=3,
                 num_resblocks=2
                 ):
        super().__init__()

        self.E = Encoder(
            c_dim,
            z_dim,
            dim,
            max_filters,
            norm,
            act,
            num_downsamples
        )

        self.D = Decoder(
            c_dim,
            max_filters,
            norm,
            act,
            num_downsamples,
            num_resblocks
        )

    def call(self, inputs, training=False):
        x, y, eps = inputs
        factors, logits = self.encode(x, training=training)
        z = self.reparameterize(factors, eps)
        x_r = self.decode(y, z, training=training)
        return x_r, factors, logits, z

    def encode(self, x, training=False):
        factors, logits = self.E(x, training=training)
        return factors, logits

    def decode(self, c, z, training=False):
        z = tf.nn.sigmoid(z)
        return self.D([c, z], training=training)

    def reparameterize(self, factors, eps):
        mu, logvar = factors
        z = mu + eps * tf.exp(0.5 * logvar)
        return z


class YVAE(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.c_dim = config['c_dim']
        self.z_dim = config['z_dim']
        dim = config['dim']
        max_filters = config['max_filters']
        norm = config['norm']
        act = config['act']
        num_downsamples = config['num_downsamples']
        num_resblocks = config['num_resblocks']

        self.G = Generator(
            self.c_dim,
            self.z_dim,
            dim,
            max_filters,
            norm,
            act,
            num_downsamples,
            num_resblocks)

        # losses
        self.lambda_ce = config['lambda_ce']
        self.lambda_zr = config['lambda_zr']
        self.beta = config['beta']

    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            eps = tf.random.normal((x.shape[0], self.z_dim))
            # reconstruction phase
            x_r, factors, logits, z = self.G([x, y, eps], True)
            factors_r, logits_r = self.G.encode(x_r, True)
            z_r = self.G.reparameterize(factors_r, eps)

            # translation phase
            y_sample = tf.random.categorical(
                tf.math.log([[1 / self.c_dim]] * self.c_dim), y.shape[0])[0]
            x_t = self.G.decode(y_sample, z, True)
            factors_t, logits_t = self.G.encode(x_t, True)
            z_t = self.G.reparameterize(factors_t, eps)

            # losses
            l_r = pix_l2_loss(x, x_r)
            l_ce = ce(y, logits)
            l_ce_t = ce(y_sample, logits_t)
            l_zr = l2_loss(z_r,z_t)
            l_kl = kl_div(z, factors)

            l = l_r + self.lambda_ce * (l_ce + l_ce_t) + self.lambda_zr * l_zr + self.beta * l_kl

        grads = tape.gradient(l, self.G.trainable_weights)
        self.optimizer[0].apply_gradients(zip(grads, self.G.trainable_weights))

        return {'l_r': l_r, 'l_ce': l_ce, 'l_ce_t': l_ce_t, 'l_zr': l_zr, 'l_kl': l_kl}

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        eps = tf.random.normal((x.shape[0], self.z_dim))
        # reconstruction phase
        x_r, factors, logits, z = self.G([x, y, eps], True)
        factors_r, logits_r = self.G.encode(x_r, True)
        z_r = self.G.reparameterize(factors_r, eps)

        # translation phase
        y_sample = tf.random.categorical(
            tf.math.log([[1 / self.c_dim]] * self.c_dim), y.shape[0])[0]
        x_t = self.G.decode(y_sample, z, True)
        factors_t, logits_t = self.G.encode(x_t, True)
        z_t = self.G.reparameterize(factors_t, eps)

        # losses
        l_r = pix_l2_loss(x, x_r)
        l_ce = ce(y, logits)
        l_ce_t = ce(y_sample, logits_t)
        l_zr = l2_loss(z_r,z_t)
        l_kl = kl_div(z, factors)
        return {'l_r': l_r, 'l_ce': l_ce, 'l_ce_t': l_ce_t, 'l_zr': l_zr, 'l_kl': l_kl}

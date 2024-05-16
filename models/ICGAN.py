import tensorflow as tf
from tensorflow.keras import layers

import sys

sys.path.append('models')

from modules import *
from losses import *

class Encoder(tf.keras.Model):
    def __init__(self, y_dim, dim=32, z_dim=100, norm='bn', act='relu', num_downsamples=4):
        super().__init__()

        self.dim = dim
        self.norm = norm
        self.act = act
        self.num_downsamples = num_downsamples

        self.Ez = self.make_net()
        self.Ey = self.make_net()

        self.to_z = tf.keras.Sequential([
            LinearBlock(4096, norm=norm, act=act),
            LinearBlock(z_dim, act='none')
        ])

        self.to_y = tf.keras.Sequential([
            LinearBlock(512, norm=norm, act=act),
            LinearBlock(y_dim, act='none')
        ])

    def make_net(self):
        net = tf.keras.Sequential([
            ConvBlock(self.dim * 2 ** i, 5, 2, norm=self.norm, act=self.act)
            for i in range(self.num_downsamples)
        ])
        net.add(layers.GlobalAveragePooling2D())
        return net

    def call(self, x, training=False):
        fz = self.Ez(x, training=training)
        fy = self.Ey(x, training=training)

        z = self.to_z(fz, training=training)
        y = self.to_y(fy, training=training)

        return z, y


class Decoder(tf.keras.Model):
    def __init__(self, nclass, dim=32, norm='bn', act='relu', num_downsamples=4):
        super().__init__()

        dim = dim * 2 ** num_downsamples
        self.blocks = tf.keras.Sequential([
            LinearBlock(4 * 4 * dim, norm, act),
            layers.Reshape([4, 4, dim])
        ])

        for i in range(num_downsamples):
            self.blocks.add(ConvBlock(dim // 2 ** i, 4, 2, norm=norm, act=act, transpose=True))

        self.blocks.add(ConvBlock(3, 4, 2, norm='none', act='sigmoid', transpose=True))
        self.nclass=nclass

    def call(self, inputs, training=False):
        z, y = inputs

        if len(y.shape)==1:
            y=tf.one_hot(y, self.nclass)

        x = tf.concat([z, y], axis=-1)
        return self.blocks(x, training=training)


class Generator(tf.keras.Model):
    def __init__(self,
                 y_dim,
                 dim=32,
                 z_dim=100,
                 norm='bn',
                 act='relu',
                 num_downsamples=4):
        super().__init__()
        self.E = Encoder(y_dim, dim, z_dim, norm, act, num_downsamples)
        self.D = Decoder(y_dim, dim, norm, act, num_downsamples)
        self.nclass=y_dim

    def call(self, x, training=False):
        z, y = self.encode(x, training=training)
        x = self.decode(z, y, training=training)
        return x

    def encode(self, x, training=False):
        return self.E(x, training=training)

    def decode(self, z, y, training=False):
        return self.D([z, y], training=training)


class Discriminator(tf.keras.Model):
    def __init__(self, nclass, dim=64, norm='bn', act='lrelu', num_downsamples=4):
        super().__init__()

        self.nclass = nclass

        self.init = tf.keras.Sequential([
            ConvBlock(dim, 4, 2, norm='none', act=act)
        ])

        self.blocks = tf.keras.Sequential()

        for i in range(1, num_downsamples):
            self.blocks.add(ConvBlock(dim * 2 ** i, 4, 2, norm=norm, act=act))

        self.blocks.add(layers.GlobalAveragePooling2D())
        self.blocks.add(LinearBlock(1, norm='none', act='none'))

    def call(self, inputs, training=False):
        x, y = inputs

        x = self.init(x)

        _, h, w, _ = x.shape

        y = tf.one_hot(y, self.nclass, axis=-1)
        y = tf.reshape(y, [-1, 1, 1, self.nclass])
        y = tf.repeat(tf.repeat(y, h, axis=1), w, axis=2)

        x = tf.concat([x, y], axis=-1)

        return self.blocks(x, training=training)


class ICGAN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.nclass = config['nclass']
        dim = config['dim']
        z_dim = config['z_dim']
        norm = config['norm']
        act = config['act']
        num_downsamples = config['num_downsamples']

        self.G = Generator(
            self.nclass,
            dim,
            z_dim,
            norm,
            act,
            num_downsamples
        )
        self.Disc = Discriminator(
            self.nclass,
            dim * 2,
            norm,
            'lrelu',
            num_downsamples)

        self.gan_loss = config['gan_loss']
        self.gp_type = config['gp_type']

    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        y_prime = tf.roll(y, 1, axis=0)

        with tf.GradientTape(persistent=True) as tape:
            xr = self.G(x, True)
            z, yr = self.G.encode(x, True)
            xt = self.G.decode(z, y_prime, True)
            zt, _ = self.G.encode(xt, True)

            real_critic = self.Disc([x, y], True)
            fake_critic = self.Disc([xt, y_prime], True)

            ##compute loss
            l_z = l2_loss(z, zt)
            l_y = l2_loss(yr, tf.one_hot(y, self.nclass))
            d_loss, g_loss, gp = adv_loss(
                real_critic,
                fake_critic,
                x,
                xt,
                self.Disc,
                self.gan_loss,
                self.gp_type,
                1,
                y
            )

            # total loss
            l_G = l_z + l_y + g_loss
            l_D = d_loss + gp

        G_grads = tape.gradient(l_G, self.G.trainable_weights)
        D_grads = tape.gradient(l_D, self.Disc.trainable_weights)

        self.optimizer[0].apply_gradients(zip(G_grads, self.G.trainable_weights))
        self.optimizer[1].apply_gradients(zip(D_grads, self.Disc.trainable_weights))

        return {'generator loss': g_loss,
                'discriminator loss': d_loss,
                'gradient penalty': gp,
                'l_z': l_z,
                'l_y': l_y,
                }

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        y_prime = tf.roll(y, 1, axis=0)

        xr = self.G(x)
        z, yr = self.G.encode(x)
        xt = self.G.decode(z, y_prime)
        zt, _ = self.G.encode(xt)

        ##compute loss
        l_z = l2_loss(z, zt)
        l_y = l2_loss(yr, tf.one_hot(y, self.nclass))

        return {'l_z': l_z,
                'l_y': l_y}




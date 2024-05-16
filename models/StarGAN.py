import tensorflow as tf
from tensorflow.keras import layers

import sys

sys.path.append('models')

from modules import *
from losses import *


class Generator(tf.keras.Model):
    def __init__(self,
                 nclass,
                 dim=64,
                 norm='in',
                 act='relu',
                 num_downsamples=3,
                 num_resblocks=6
                 ):
        super().__init__()

        self.blocks = tf.keras.Sequential([
            ConvBlock(dim, 7, 1, 'same', norm, act)
        ])

        for i in range(1, num_downsamples):
            self.blocks.add(
                ConvBlock(dim * 2 ** i, 4, 2, 'same', norm, act)
            )

        for _ in range(num_resblocks):
            self.blocks.add(
                ResBlock(dim * 2 ** i, norm, act)
            )

        dim = dim * 2 ** i
        for i in range(1, num_downsamples):
            self.blocks.add(layers.UpSampling2D())
            self.blocks.add(ConvBlock(dim // (2 ** i), 5, 1, 'same', 'in', act))

        self.blocks.add(ConvBlock(3, 7, 1, 'same', act='sigmoid'))

        self.nclass = nclass

    def call(self, inputs, training=False):
        x, c = inputs
        _, h, w, _ = x.shape

        c = tf.one_hot(c, self.nclass, axis=-1)
        c = tf.reshape(c, [-1, 1, 1, self.nclass])
        c = tf.repeat(tf.repeat(c, h, axis=1), w, axis=2)

        x = tf.concat([x, c], axis=-1)

        return self.blocks(x, training=training)


class Discriminator(tf.keras.Model):
    def __init__(self,
                 nclass,
                 dim=64,
                 num_downsamples=6,
                 norm='in'):
        super().__init__()

        self.blocks = tf.keras.Sequential([
            ConvBlock(dim * 2 ** i, 4, 2, 'same', norm, act='lrelu')
            for i in range(num_downsamples)
        ])

        self.to_critic = ConvBlock(1, 3, 1, 'same', act='none')
        self.nclass = nclass

    def build(self, shape):
        _, h, w, c = shape
        self.to_cls = ConvBlock(self.nclass, h // 64, 1, 'valid', 'none', act='none')

    def call(self, x, training=False):
        x = self.blocks(x, training=training)
        critic = self.to_critic(x)
        logits = self.to_cls(x)
        return critic, logits[:, 0, 0, :]


class StarGAN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.nclass = config['nclass']
        dim = config['dim']
        norm = config['norm']
        act = config['act']
        num_downsamples = config['num_downsamples']
        num_resblocks = config['num_resblocks']
        self.gan_loss = config['gan_loss']
        self.gp_type = config['gp_type']
        self.lambda_cls = config['lambda_cls']
        self.lambda_r = config['lambda_r']

        self.G = Generator(
            nclass=self.nclass,
            dim=dim,
            norm=norm,
            act=act,
            num_downsamples=num_downsamples,
            num_resblocks=num_resblocks
        )
        self.Disc = Discriminator(
            nclass=self.nclass,
            dim=dim,
            num_downsamples=6,
            norm='none' if self.gan_loss == 'wgan' else norm
        )

    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        y_prime = tf.roll(y, 1, axis=0)
        with tf.GradientTape(persistent=True) as tape:
            # forward
            x_t = self.G([x, y_prime], True)
            x_r = self.G([x_t, y], True)

            real_critic, real_logits = self.Disc(x, True)
            fake_critic, fake_logits = self.Disc(x_t, True)

            # compute loss

            l_r = l1_loss(x, x_r)

            d_loss, g_loss, gp = adv_loss(
                real_critic,
                fake_critic,
                x,
                x_t,
                self.Disc,
                self.gan_loss,
                self.gp_type)

            l_c_d = ce(y, real_logits)
            l_c_g = ce(y_prime, fake_logits)

            l_G = g_loss + self.lambda_cls * l_c_g + self.lambda_r * l_r
            l_D = d_loss + self.lambda_cls * l_c_d +  10 * gp

        G_grads = tape.gradient(l_G, self.G.trainable_weights)
        D_grads = tape.gradient(l_D, self.Disc.trainable_weights)

        self.optimizer[0].apply_gradients(zip(G_grads, self.G.trainable_weights))
        self.optimizer[1].apply_gradients(zip(D_grads, self.Disc.trainable_weights))

        return {'reconstruction loss': l_r,
                'generator loss': g_loss,
                'discriminator loss': d_loss,
                'gradient penalty': gp,
                'G cross entropy': l_c_g,
                'D cross entropy': l_c_d,
                }

    @tf.function
    def test_step(self, inputs):
        x, y = inputs

        y_prime = tf.roll(y, 1, axis=0)
        # forward
        x_t = self.G([x, y_prime], True)
        x_r = self.G([x_t, y], True)

        # compute loss
        l_r = l1_loss(x, x_r)

        real_critic, real_logits = self.Disc(x, True)
        fake_critic, fake_logits = self.Disc(x_t, True)

        l_c_d = ce(y, real_logits)
        l_c_g = ce(y_prime, fake_logits)

        return {'reconstruction loss': l_r,
                'G cross entropy': l_c_g,
                'D cross entropy': l_c_d,
                }




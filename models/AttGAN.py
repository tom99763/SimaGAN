import tensorflow as tf
from tensorflow.keras import layers

import sys

sys.path.append('../input/toolss/Multimodal-Contrastive-Translation-main/models')

from modules import *
from losses import *


class Generator(tf.keras.Model):
    def __init__(self, nclass, dim=64, norm='bn', act='lrelu', num_downsamples=5):
        super().__init__()

        self.E = [
            ConvBlock(dim * 2 ** i, 4, 2, norm=norm, act=act)
            for i in range(num_downsamples)
        ]

        dim = dim * 2 ** num_downsamples - 1
        self.D = [
            ConvBlock(dim // 2 ** i, 4, 2, norm=norm, act=act, transpose=True)
            for i in range(num_downsamples - 1)
        ]

        self.to_rgb = ConvBlock(3, 4, 2, norm='none', act='sigmoid', transpose=True)

        self.nclass = nclass

    def call(self, inputs, training=False):
        x, y = inputs
        z, fmaps = self.encode(x, training=training)
        x = self.decode(z, y, fmaps, training=training)
        return x, z, fmaps

    def encode(self, x, training=False):
        fmaps = []
        for i, block in enumerate(self.E):
            x = block(x, training=training)
            if i != len(self.E) - 1:
                fmaps.append(x)
        return x, fmaps

    def decode(self, x, y, fmaps, training=False):
        b, h, w, _ = x.shape
        y = tf.one_hot(y, self.nclass)
        y = tf.reshape(y, [-1, 1, 1, self.nclass])
        y = tf.repeat(tf.repeat(y, h, axis=1), w, axis=2)
        x = tf.concat([x, y], axis=-1)

        for i, block in enumerate(self.D):
            x = block(x, training=training)
            x = tf.concat([x, fmaps[-i - 1]], axis=-1)
        x = self.to_rgb(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, nclass, dim=64, norm='in', act='lrelu', num_downsamples=5):
        super().__init__()

        self.nclass = nclass

        self.blocks = tf.keras.Sequential()

        for i in range(num_downsamples):
            self.blocks.add(ConvBlock(dim * 2 ** i, 4, 2, norm=norm, act=act))

        self.blocks.add(layers.GlobalAveragePooling2D())

        self.to_critic = tf.keras.Sequential([
            LinearBlock(1024, norm, act),
            LinearBlock(1, 'none', 'none')
        ])

        self.to_logits = tf.keras.Sequential([
            LinearBlock(1024, norm, act),
            LinearBlock(nclass, 'none', 'none')
        ])

    def call(self, x):
        x = self.blocks(x)
        critic = self.to_critic(x)
        logits = self.to_logits(x)
        return critic, logits


class AttGAN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.nclass = config['nclass']
        dim = config['dim']
        norm = config['norm']
        act = config['act']
        num_downsamples = config['num_downsamples']
        self.lambda_r = config['lambda_r']
        self.lambda_cls_g = config['lambda_cls_g']
        self.lambda_cls_d = config['lambda_cls_d']
        self.gan_loss = config['gan_loss']
        self.gp_type = config['gp_type']

        self.G = Generator(
            self.nclass,
            dim,
            norm,
            act,
            num_downsamples
        )
        self.Disc = Discriminator(
            self.nclass,
            dim,
            'none' if self.gp_type == 'wgan' else 'in',
            'lrelu',
            num_downsamples)


    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        y_prime = tf.roll(y, 1, axis=0)

        with tf.GradientTape(persistent=True) as tape:
            xr, zr, fmapsr = self.G([x, y], True)
            xt = self.G.decode(zr, y_prime, fmapsr, True)

            real_critic, real_logits = self.Disc(x)
            fake_critic, fake_logits = self.Disc(xt)

            ### compute loss
            l_r = l1_loss(x, xr)

            d_loss, g_loss, gp = adv_loss(
                real_critic,
                fake_critic,
                x,
                xt,
                self.Disc,
                self.gan_loss,
                self.gp_type,
                2
            )

            l_cls_d = ce(y, real_logits)
            l_cls_g = ce(y_prime, fake_logits)

            l_G = self.lambda_r * l_r + g_loss + self.lambda_cls_g * l_cls_g
            l_D = d_loss + gp + self.lambda_cls_d * l_cls_d

        G_grads = tape.gradient(l_G, self.G.trainable_weights)
        D_grads = tape.gradient(l_D, self.Disc.trainable_weights)

        self.optimizer[0].apply_gradients(zip(G_grads, self.G.trainable_weights))
        self.optimizer[1].apply_gradients(zip(D_grads, self.Disc.trainable_weights))
        return {'g_loss': g_loss, 'd_loss': d_loss, 'l_r': l_r, 'l_cls_g': l_cls_g, 'l_cls_d': l_cls_d}

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        y_prime = tf.roll(y, 1, axis=0)
        xr, zr, fmapsr = self.G([x, y])
        xt = self.G.decode(zr, y_prime, fmapsr)

        real_critic, real_logits = self.Disc(x)
        fake_critic, fake_logits = self.Disc(xt)

        ### compute loss
        l_r = l1_loss(x, xr)
        l_cls_d = ce(y, real_logits)
        l_cls_g = ce(y_prime, fake_logits)
        return {'l_r': l_r, 'l_cls_g': l_cls_g, 'l_cls_d': l_cls_d}

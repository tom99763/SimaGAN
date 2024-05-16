import tensorflow as tf
from tensorflow.keras import layers

import sys

sys.path.append('../input/toolss/Multimodal-Contrastive-Translation-main/models')

from modules import *
from losses import *


class Encoder(tf.keras.Model):
    def __init__(self, nclass, dim=64, norm='l2', act='lrelu', num_downsamples=5, max_filters=512):
        super().__init__()

        self.blocks = [
            ConvBlock(min(dim * 2 ** i, max_filters), 3, 2, norm=norm, act=act)
            for i in range(num_downsamples)
        ]

        self.to_z = LinearBlock(nclass, norm=norm, act='none')

    def call(self, x):
        fmaps = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i != len(self.blocks) - 1:
                fmaps.append(x)
        z = self.to_z(x)
        return z, fmaps


class Decoder(tf.keras.Model):
    def __init__(self, dim=64, norm='l2', act='lrelu', num_downsamples=5, max_filters=512):
        super().__init__()

        dim = dim * 2 ** num_downsamples - 1
        self.blocks = [
            ConvBlock(dim // 2 ** i, 3, 2, norm=norm, act=act, transpose=True)
            for i in range(num_downsamples - 1)
        ]

        self.to_rgb = ConvBlock(3, 3, 2, norm='none', act='tanh', transpose=True)

    def call(self, inputs):
        x, z1, z2, fmaps = inputs
        z = tf.concat([z1, z2], axis=-1)
        for i, block in enumerate(self.blocks):
            z = block(z)
            z = tf.concat([z, fmaps[-i - 1]], axis=-1)
        r = self.to_rgb(z)
        x = tf.clip_by_value(x + 2 * r, -1, 1) * 0.5 + 0.5
        return x


class Generator(tf.keras.Model):
    def __init__(self, nclass, dim=64, norm='l2', act='lrelu', num_downsamples=5):
        super().__init__()

        self.E = Encoder(nclass, dim, norm, act, num_downsamples)
        self.D = Decoder(dim, norm, act, num_downsamples)

        self.nclass = nclass

    def call(self, inputs):
        x, y = inputs
        z, fmaps = self.encode(x)
        x = self.decode(x, z, z, y, y, fmaps)
        return x, z, fmaps

    def encode(self, x):
        return self.E(x)

    def decode(self, x, z1, z2, y1, y2, fmaps):
        z12 = self.swap(z1, z2, y1, y2)
        return self.D([x, z12, z1, fmaps])

    def get_idx(self, z, y):
        s = [tf.range(z.shape[i]) for i in range(3)]
        d1, d2, d3 = tf.meshgrid(s[1], s[0], s[2])
        idx = tf.stack([d2, d1, d3], axis=-1)
        idx = tf.cast(idx ,'int64')
        _, h, w, _ = idx.shape
        y = tf.repeat(tf.repeat(y[:, None, None], h, axis=1), w, axis=2)[..., None]
        idx = tf.concat([idx, y], axis=-1)
        return idx

    def get_corr_ele(self, z, y1, y2):
        idx1 = self.get_idx(z, y1)
        idx2 = self.get_idx(z, y2)
        idx = tf.concat([idx1, idx2], axis=0)
        ele = tf.gather_nd(z, idx)
        return ele, idx

    def swap(self, z1, z2, y1, y2):
        z1y, idx = self.get_corr_ele(z1, y1, y2)
        z2y, _ = self.get_corr_ele(z2, y1, y2)
        z12 = tf.tensor_scatter_nd_update(z1, idx, z2y)
        # z21 = tf.tensor_scatter_nd_update(z2, idx, x1y)

        # straight throguh estimator
        z12 = z1 + tf.stop_gradient(z12 - z1)
        # z21 = z2 + tf.stop_gradient(z21-z2)
        return z12  # , z21


class Discriminator(tf.keras.Model):
    def __init__(self, nclass, image_size=128, dim=64, norm='l2', act='lrelu', num_downsamples=5):
        super().__init__()

        self.nclass = nclass
        self.image_size = image_size

        self.blocks = tf.keras.Sequential()

        for i in range(num_downsamples - 1):
            self.blocks.add(ConvBlock(dim * 2 ** i, 3, 2, norm=norm, act=act))

        self.downsample = layers.AveragePooling2D()

        self.to_critic = tf.keras.Sequential([
            layers.Flatten(),
            LinearBlock(1, act='none')
        ])

    def call(self, inputs):
        x, y = inputs

        if self.image_size != x.shape[1]:
            x = self.downsample(x)

        _, h, w, _ = x.shape
        y = tf.one_hot(y, self.nclass, axis=-1)
        y = tf.reshape(y, [-1, 1, 1, self.nclass])
        y = tf.repeat(tf.repeat(y, h, axis=1), w, axis=2)
        x = tf.concat([x, y], axis=-1)
        x = self.blocks(x)
        critic = self.to_critic(x)
        return critic


class ELEGANT(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.nclass = config['nclass']
        dim = config['dim']
        norm = config['norm']
        act = config['act']
        num_downsamples = config['num_downsamples']
        self.gan_loss = config['gan_loss']
        self.gp_type = config['gp_type']

        self.G = Generator(
            self.nclass,
            dim,
            norm,
            act,
            num_downsamples
        )
        self.Disc1 = Discriminator(
            self.nclass,
            128,
            dim,
            'none ' if self.gp_type == 'wgan' else 'l2',
            'lrelu',
            num_downsamples)

        self.Disc2 = Discriminator(
            self.nclass,
            64,
            dim,
            'none ' if self.gp_type == 'wgan' else 'l2',
            'lrelu',
            num_downsamples)


    @tf.function
    def train_step(self, inputs):
        x, y = inputs
        x_prime = tf.roll(x, 1, axis=0)
        y_prime = tf.roll(y, 1, axis=0)

        with tf.GradientTape(persistent=True) as tape:
            xr, z, fmaps = self.G([x, y])
            zt, _ = self.G.encode(x_prime)
            xt = self.G.decode(x, z, zt, y, y_prime, fmaps)

            real_critic1 = self.Disc1([x, y])
            real_critic2 = self.Disc2([x, y])
            fake_critic1 = self.Disc1([xt, y_prime])
            fake_critic2 = self.Disc2([xt, y_prime])

            l_r = l1_loss(x, xr)

            d_loss1, g_loss1, gp1 = adv_loss(
                real_critic1,
                fake_critic1,
                x,
                xt,
                self.Disc1,
                self.gan_loss,
                self.gp_type,
                1,
                y=y
            )

            d_loss2, g_loss2, gp2 = adv_loss(
                real_critic2,
                fake_critic2,
                x,
                xt,
                self.Disc2,
                self.gan_loss,
                self.gp_type,
                1,
                y=y
            )

            l_G = l_r + g_loss1 + g_loss2
            l_D = d_loss1 + d_loss2 + gp1 + gp2

        G_grads = tape.gradient(l_G, self.G.trainable_weights)
        D_grads = tape.gradient(l_D, self.Disc1.trainable_weights + self.Disc2.trainable_weights)

        self.optimizer[0].apply_gradients(zip(G_grads, self.G.trainable_weights))
        self.optimizer[1].apply_gradients(zip(D_grads, self.Disc1.trainable_weights + self.Disc2.trainable_weights))

        return {'l_r': l_r,
                'g_loss1': g_loss1,
                'g_loss2': g_loss2,
                'd_loss1': d_loss1,
                'd_loss2': d_loss2,
                'gp1': gp1,
                'gp2': gp2,
                }

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        xr, z, fmaps = self.G([x, y])
        l_r = l1_loss(x, xr)
        return {'l_r': l_r}

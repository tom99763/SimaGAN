import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

import sys

sys.path.append('models')
from modules import *
from losses import *


def kldiv(x, prior):
    sample = prior.sample(1000)
    empirical = tfp.distributions.Normal(
        tf.reduce_mean(x, axis=0),
        tf.math.reduce_variance(x, axis=0))
    return tf.reduce_mean(-empirical.log_prob(sample) - prior.entropy())


class Encoder(tf.keras.Model):
    def __init__(self, dim=64, c_dim=32, s_dim=128, max_filters=512, num_downsamples=3, norm='in', act='lrelu'):
        super().__init__()

        self.blocks = tf.keras.Sequential([
            ConvBlock(dim, 1, 1, 'valid', 'none', act)
        ])

        for i in range(1, num_downsamples + 1):
            filters = min(dim * 2 ** i, max_filters)
            self.blocks.add(ResBlock(filters, norm, act, downsample=True, reflect=True))

        self.to_c = tf.keras.Sequential([
            ResBlock(c_dim, norm, act, reflect=True) for i in range(2)
        ])

        self.to_s = tf.keras.Sequential([
            ConvBlock(max_filters, 3, 2, 'valid', 'none', act),
            ConvBlock(max_filters, 3, 2, 'valid', 'none', act),
            layers.GlobalAveragePooling2D(),
            LinearBlock(s_dim, act='none')
        ])

    def call(self, x):
        x = self.blocks(x)
        c = self.to_c(x)
        s = self.to_s(x)
        return c, s


class Decoder(tf.keras.Model):
    def __init__(self, dim=32, norm='in', act='lrelu'):
        super().__init__()
        if dim == 32:
            ch_multiplier = (16, 16, 16, 8, 4, 2)
        else:
            ch_multiplier = (4, 4, 4, 4, 2, 1)
        upsample = (False, False, False, True, True, True)
        self.blocks = []
        for mul, up in zip(ch_multiplier, upsample):
            filters = dim * mul
            self.blocks.append(ResBlock(filters, 'adain', act, upsample=up))

        self.to_rgb = tf.keras.Sequential([
            ConvBlock(3, 1, 1, 'same', 'none', 'none'),
            layers.Lambda(lambda x: 0.5 * x + 0.5)
        ])

    def call(self, inputs):
        x, w = inputs
        for block in self.blocks:
            x = block([x, w])
        x = self.to_rgb(x)
        return x


class Generator(tf.keras.Model):
    def __init__(self,
                 dim=32,
                 c_dim=32, #16, 32, 64
                 s_dim=128, #64, 128, 256
                 max_filters=512,
                 num_downsamples=3,
                 norm='in',
                 act='lrelu'):
        super().__init__()
        self.E = Encoder(dim, c_dim, s_dim, max_filters, num_downsamples, norm, act)
        #self.E = backbone  output: style (1d bxd=128) & content(2d  bxhxwxd=32)
        self.D = Decoder(dim, norm, act)

        self.mlp = tf.keras.Sequential([LinearBlock(max_filters//2, act='lrelu') for _ in range(4)])
        self.mlp.add(LinearBlock(s_dim, act='none'))

    def call(self, x):
        c, s = self.encode(x)
        x = self.decode(c, s)
        return x, c, s

    def encode(self, x):
        c, s = self.E(x)
        return c, s

    def decode(self, c, s):
        s = self.mlp(s)
        return self.D([c, s])


class Discriminator(tf.keras.Model):
    def __init__(self, nclass, dim=64, q_dim=128, max_filters=512, norm='in', act='lrelu', num_downsamples=5):
        super().__init__()

        self.blocks = tf.keras.Sequential([
            ConvBlock(dim, 1, 1, 'valid', 'none', act)
        ])

        for i in range(1, num_downsamples + 1):
            filters = min(dim * 2 ** i, max_filters)
            self.blocks.add(ResBlock(filters, norm, act, downsample=True, reflect=True))

        self.blocks.add(ConvBlock(max_filters, 3, 1, act=act))
        self.blocks.add(layers.Flatten())

        self.to_critic = tf.keras.Sequential([
            LinearBlock(max_filters, act=act),
            LinearBlock(1, act='none')
        ])
        self.to_q = LinearBlock(q_dim, act='none')
        self.to_e = layers.Embedding(nclass, q_dim)

    def call(self, inputs):
        x, y = inputs
        x = self.blocks(x)
        critic = self.to_critic(x)
        q = self.to_q(x)
        e = self.to_e(y)
        return critic, q, e


class Latent_Discriminator(tf.keras.Model):
    def __init__(self, dim):
        super().__init__()
        self.D = tf.keras.Sequential([
            LinearBlock(dim, act='lrelu'),
            LinearBlock(dim, act='lrelu')
        ])
        self.to_critic = LinearBlock(1, act='none')

    def call(self, x):
        f = self.D(x)
        critic = self.to_critic(f)
        return critic


class SimaGAN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.nclass = config['nclass']
        dim = config['dim']
        c_dim = config['c_dim']
        q_dim = config['q_dim']
        self.s_dim = config['s_dim']
        max_filters = config['max_filters']
        num_downsamples = config['num_downsamples']
        norm = config['norm']
        act = config['act']

        self.G = Generator(
            dim,
            c_dim,
            self.s_dim,
            max_filters,
            num_downsamples,
            norm,
            act)

        self.Disc = Discriminator(
            self.nclass,
            dim=dim,
            q_dim=q_dim,
            max_filters=max_filters,
            norm=norm,
            act=act
        )

        self.LDisc = Latent_Discriminator(max_filters)

        self.prior = tfp.distributions.Normal(0., [1.] * self.s_dim)

        self.lambda_r = config['lambda_r']
        self.lambda_dce = config['lambda_dce']
        self.gan_loss = config['gan_loss']
        self.gp_type = config['gp_type']
        self.tau = config['tau']

    @tf.function
    def train_step(self, inputs):
        x, y = inputs

        with tf.GradientTape(persistent=True) as tape:
            # reconstruction
            xr, c, z = self.G(x)

            # translation
            ca, cb = tf.split(c, 2, axis=0)
            za, zb = tf.split(z, 2, axis=0)
            xab = self.G.decode(ca, zb)
            xba = self.G.decode(cb, za)
            x_fake = tf.concat([xba, xab], axis=0)

            # discrimination
            critic_real, q_real, e = self.Disc([x, y])
            critic_fake, q_fake, _ = self.Disc([x_fake, y])

            # sampling & latent discrimination
            z_prior = self.prior.sample(z.shape[0])
            critic_real_l = self.LDisc(z_prior)
            critic_fake_l = self.LDisc(z)

            ####compute losses
            # reconstruction
            l_r = wavelet_dis(x, xr, 3)

            # adversarial loss
            d_loss, g_loss, gp = adv_loss(
                critic_real, critic_fake,
                x, xr, self.Disc,
                self.gan_loss, self.gp_type, 3, y)

            # latent adversarial loss
            d_loss_l = bce(tf.ones_like(critic_real_l), critic_real_l) + \
                       bce(tf.zeros_like(critic_fake_l), critic_fake_l)
            g_loss_l = bce(tf.ones_like(critic_fake_l), critic_fake_l)

            # conditional contrastive loss
            l_dce_g = infoDCE(q_fake, e, y, self.tau)
            l_dce_d = infoDCE(q_real, e, y, self.tau)

            # total loss
            l_G = g_loss + g_loss_l + self.lambda_r * l_r + self.lambda_dce * l_dce_g
            l_D = d_loss + d_loss_l + 10 * gp + self.lambda_dce * l_dce_d

        G_grads = tape.gradient(l_G, self.G.trainable_weights)
        D_grads = tape.gradient(l_D, self.Disc.trainable_weights + \
                                self.LDisc.trainable_weights)

        self.optimizer[0].apply_gradients(zip(G_grads, self.G.trainable_weights))
        self.optimizer[1].apply_gradients(zip(D_grads, self.Disc.trainable_weights + \
                                              self.LDisc.trainable_weights))

        # evaluate kl-divergence
        kl = kldiv(z, self.prior)

        return {'l_r': l_r,
                'g_loss': g_loss,
                'd_loss': d_loss,
                'gp': gp,
                'g_loss_l': g_loss_l,
                'd_loss_l': d_loss_l,
                'l_dce_g': l_dce_g,
                'l_dce_d': l_dce_d,
                'kl': kl
                }

    @tf.function
    def test_step(self, inputs):
        x, y = inputs
        # reconstruction
        xr, c, z = self.G(x)

        # translation
        ca, cb = tf.split(c, 2, axis=0)
        za, zb = tf.split(z, 2, axis=0)
        xab = self.G.decode(ca, zb)
        xba = self.G.decode(cb, za)
        x_fake = tf.concat([xba, xab], axis=0)

        # discrimination
        critic_real, q_real, e = self.Disc([x, y])
        critic_fake, q_fake, _ = self.Disc([x_fake, y])

        ####compute losses
        # reconstruction
        l_r = wavelet_dis(x, xr, 3)

        # conditional contrastive loss
        l_dce_g = infoDCE(q_fake, e, y, self.tau)
        l_dce_d = infoDCE(q_real, e, y, self.tau)

        # evaluate kl-divergence
        kl = kldiv(z, self.prior)

        return {'l_r': l_r,
                'l_dce_g': l_dce_g,
                'l_dce_d': l_dce_d,
                'kl': kl
                }
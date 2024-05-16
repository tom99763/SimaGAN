import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers


# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))


# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)


class PPLSampler:
    def __init__(self, model, epsilon=1e-4, sampling='full', space='z'):
        self.model = model
        self.epsilon = epsilon
        self.vgg16 = self.build_vgg16()
        self.sampling = sampling
        self.space = space

    def build_vgg16(self):
        inputs = layers.Input(shape=(224, 224, 3), name='input')
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
        outputs = vgg16.get_layer('block4_conv3').output
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def preprocess(self, x):
        x *= 255.
        x = tf.image.resize(x, (224, 224))
        x = preprocess_input(x)
        return x

    def __call__(self, x):
        # encode content
        f_content, _ = self.model.G.encode(x)

        # Generate random latents and interpolation t-values.
        t = tf.random.uniform([f_content.shape[0]//2, 1], 0., 1.) * (1. if self.sampling == 'full' else 0.)
        z = tf.random.normal([f_content.shape[0] * 1, self.model.s_dim])

        # Interpolate in W or Z.
        if self.space == 'w':
            w = self.model.G.mlp(z)
            w0, w1 = tf.split(w, 2, axis=0)
            wt0 = w0 + t * (w1 - w0)
            wt1 = w0 + (t + self.epsilon) * (w1 - w0)
            wt = tf.concat([wt0, wt1], axis=0)

        else:
            z0, z1 = tf.split(z, 2, axis=0)
            zt0 = slerp(z0, z1, t)
            zt1 = slerp(z0, z1, t + self.epsilon)
            wt = self.model.G.mlp(tf.concat([zt0, zt1], axis=0))

        # decode image
        img = self.model.G.D([f_content, wt])

        # compute lpips
        lpips_t0, lpips_t1 = tf.split(self.vgg16(self.preprocess(img)), 2, axis=0)
        dist = tf.reduce_sum((lpips_t0 - lpips_t1) ** 2, axis=-1)* self.epsilon
        return dist


def compute_ppl(ds, model, space, sampling):
    sampler = PPLSampler(model, 1e-4, sampling, space)
    dist = []
    for x, y in ds:
        for i in range(10):
            dist.append(sampler(x))
    dist = tf.concat(dist, axis=0).numpy()
    lo = np.percentile(dist, 1, interpolation='lower')
    hi = np.percentile(dist, 99, interpolation='higher')
    ppl = np.extract(np.logical_and(dist >= lo, dist <= hi), dist).mean()
    return float(ppl)
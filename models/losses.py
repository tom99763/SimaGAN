import tensorflow as tf
from tensorflow.keras import losses
import math
import numpy as np
import pywt

ce = losses.SparseCategoricalCrossentropy(from_logits=True)
bce = losses.BinaryCrossentropy(from_logits=True)


##### Distances
def l1_loss(x1, x2):
    return tf.reduce_mean(tf.abs(x1 - x2))


def l2_loss(x1, x2):
    return tf.reduce_mean(tf.square(x1 - x2))


def pix_l2_loss(x1, x2):
    return tf.reduce_mean(tf.math.reduce_sum(tf.math.square(x1 - x2), axis=[1, 2, 3]))


def frobenius(x):
    return tf.reduce_mean(tf.square(x), axis=[1, 2])


# coefficient wise loss
def tf_dwt(yl, wave='haar'):
    w = pywt.Wavelet(wave)
    ll = np.outer(w.dec_lo, w.dec_lo)
    lh = np.outer(w.dec_hi, w.dec_lo)
    hl = np.outer(w.dec_lo, w.dec_hi)
    hh = np.outer(w.dec_hi, w.dec_hi)
    d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    d_temp[::-1, ::-1, 0, 0] = ll
    d_temp[::-1, ::-1, 0, 1] = lh
    d_temp[::-1, ::-1, 0, 2] = hl
    d_temp[::-1, ::-1, 0, 3] = hh

    filts = d_temp.astype('float32')

    filts = filts[None, :, :, :, :]

    filter = tf.convert_to_tensor(filts)
    sz = 2 * (len(w.dec_lo) // 2 - 1)

    yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz], [sz, sz], [0, 0]]), mode='reflect')

    y = tf.expand_dims(yl, 1)
    inputs = tf.split(y, [1] * int(y.shape.dims[4]), 4)
    inputs = tf.concat([x for x in inputs], 1)

    outputs_3d = tf.nn.conv3d(inputs, filter, padding='VALID', strides=[1, 1, 2, 2, 1])
    outputs = tf.split(outputs_3d, [1] * int(outputs_3d.shape.dims[1]), 1)
    outputs = tf.concat([x for x in outputs], 4)

    outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2],
                                   tf.shape(outputs)[3], tf.shape(outputs)[4]))

    return outputs


def tf_idwt(y, wave='haar'):
    w = pywt.Wavelet(wave)
    ll = np.outer(w.rec_lo, w.rec_lo)
    lh = np.outer(w.rec_hi, w.rec_lo)
    hl = np.outer(w.rec_lo, w.rec_hi)
    hh = np.outer(w.rec_hi, w.rec_hi)
    d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    d_temp[:, :, 0, 0] = ll
    d_temp[:, :, 0, 1] = lh
    d_temp[:, :, 0, 2] = hl
    d_temp[:, :, 0, 3] = hh
    filts = d_temp.astype('float32')
    filts = filts[None, :, :, :, :]
    filter = tf.convert_to_tensor(filts)
    s = 2 * (len(w.dec_lo) // 2 - 1)
    out_size = tf.shape(y)[1]

    y = tf.expand_dims(y, 1)
    inputs = tf.split(y, [4] * int(int(y.shape.dims[4]) / 4), 4)
    inputs = tf.concat([x for x in inputs], 1)

    outputs_3d = tf.nn.conv3d_transpose(inputs, filter, output_shape=[tf.shape(y)[0], tf.shape(inputs)[1],
                                                                      2 * (out_size - 1) + np.shape(ll)[0],
                                                                      2 * (out_size - 1) + np.shape(ll)[0], 1],
                                        padding='VALID', strides=[1, 1, 2, 2, 1])
    outputs = tf.split(outputs_3d, [1] * int(int(y.shape.dims[4]) / 4), 1)
    outputs = tf.concat([x for x in outputs], 4)

    outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2],
                                   tf.shape(outputs)[3], tf.shape(outputs)[4]))
    outputs = outputs[:, s: 2 * (out_size - 1) + np.shape(ll)[0] - s, s: 2 * (out_size - 1) + np.shape(ll)[0] - s,
              :]
    return outputs


def wavelet_content_texture_loss(cx, cy):
    l_content = 0.
    for i in range(12):
        l_content = l_content + l2_loss(cx[..., i], cy[..., i])

    l_texture = 0.
    cx_ = tf.gather(cx, [1, 2, 3, 5, 6, 7, 9, 10, 11], axis=-1)
    cy_ = tf.gather(cy, [1, 2, 3, 5, 6, 7, 9, 10, 11], axis=-1)
    for i in range(9):
        l_texture += tf.reduce_mean(
            tf.nn.relu(1.2 * frobenius(cx_[..., i]) \
                       - frobenius(cy_[..., i])))
    return l_content + l_texture


def wavelet_dis(x, y, num_levels=3):
    xll = x
    yll = y
    loss = 0.
    for i in range(num_levels):
        cx = tf_dwt(xll)
        cy = tf_dwt(yll)
        loss = loss + wavelet_content_texture_loss(cx, cy)
        xll = tf.gather(cx, [0, 4, 8], axis=-1)
        yll = tf.gather(cy, [0, 4, 8], axis=-1)
    return loss


##### Probability
def kl_div(mu, logvar, prior="normal"):
    if prior.lower() == "normal":
        summand = mu ** 2 + tf.exp(logvar) - logvar - 1
        return tf.reduce_mean(0.5 * tf.math.reduce_sum(summand, [1]), name="kl_loss")

    if prior.lower() == "laplace":
        exponent = 0.5 * logvar - tf.abs(mu) * tf.exp(- 0.5 * logvar)
        summand = tf.abs(mu) + tf.exp(exponent) - 0.5 * logvar
        return tf.reduce_mean(summand, [1], name="kl_loss")


def tc_penalty(z, mu, logvar, prior="normal"):
    tc = total_correlation(z, mu, logvar, prior)
    return tc


def gaussian_log_density(z, mu, logvar):
    pi = tf.constant(math.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-logvar)
    tmp = (z - mu)
    return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)


def laplace_log_density(z, mu, logvar):
    c = tf.math.log(0.5)
    tmp = tf.math.abs(z - mu)
    return c - 0.5 * logvar - tf.math.exp(-0.5 * logvar) * tmp


def total_correlation(z, mu, logvar, prior):
    if prior.lower() == "normal":
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(mu, 0),
            tf.expand_dims(logvar, 0))
    if prior.lower() == "laplace":
        log_qz_prob = laplace_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(mu, 0),
            tf.expand_dims(logvar, 0))
    log_qz_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False), axis=1, keepdims=False)
    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_qz_prob, axis=2, keepdims=False), axis=1, keepdims=False)
    return tf.math.reduce_mean(log_qz - log_qz_product)


##### Mutual Information bounds
def infoNCE(f_q, f_k, tau=0.07):
    '''
    f_q: (b,d)  from  Q(Enc_s(Dec(c,shift(s))))
    f_k: (b,d), from  shift(Q(s))
    '''
    b, d = f_q.shape

    f_q = tf.math.l2_normalize(f_q, axis=-1)
    f_k = tf.math.l2_normalize(f_k, axis=-1)

    # positive
    l_pos = tf.reduce_sum(f_q * f_k, axis=-1, keepdims=True)  # (b,1)
    # negative
    y = tf.eye(b)
    mask = tf.where(y == 1, -float('inf'), y)
    l_neg = f_q @ tf.transpose(f_k)  # (b,b)
    l_neg = mask + l_neg

    # compute loss
    logits = tf.concat([l_pos, l_neg], axis=-1) / tau  # (b,b+1)
    targets = tf.zeros((b,))
    loss = ce(targets, logits)
    # loss=tf.reduce_mean(
    # -tf.math.log(tf.exp(logits[:,0])/\
    # tf.reduce_sum(tf.exp(logits),axis=-1)))
    return loss


def infoDCE(f, e, y, tau=0.07):
    b, d = f.shape
    f = tf.math.l2_normalize(f, axis=-1)
    e = tf.math.l2_normalize(e, axis=-1)
    l_pos = tf.reduce_sum(f * e, axis=-1, keepdims=True)  # (b,1)

    y = tf.cast(y, 'int32')
    y = tf.one_hot(y, tf.reduce_max(y) + 1, axis=-1)
    y = y @ tf.transpose(y)
    mask = tf.where(y == 1, -float('inf'), y)
    l_neg = f @ tf.transpose(f) + mask
    logits = tf.concat([l_pos, l_neg], axis=-1) / tau  # (b,b+1)
    targets = tf.zeros((b,))
    loss = ce(targets, logits)
    return loss



#####GANs
def gradient_penalty(real_image, fake_image, disc, y=None):
    if y is None:
        y = tf.constant([0] * real_image.shape[0])
    b, _, _, _ = real_image.shape
    epsilon = tf.random.uniform(shape=[b, 1, 1, 1], minval=0., maxval=1.)
    y = tf.constant([0] * real_image.shape[0])
    with tf.GradientTape() as tape:
        mixed_image = epsilon * real_image + (1 - epsilon) * fake_image
        tape.watch(mixed_image)
        try:
            mixed_critic, _, _ = disc([mixed_image, y])
        except:
            mixed_critic, _ = disc(mixed_image)
    grads = tape.gradient(mixed_critic, mixed_image)
    grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean(tf.square(grad_norms - 1))
    return gp


def r1_regularzation(real_image, disc, output=1, y=None):
    if y is None:
        use_y = False
    else:
        use_y = True
    with tf.GradientTape() as tape:
        tape.watch(real_image)

        if output == 1:
            critic = disc([real_image, y]) if use_y else disc(real_image)

        elif output == 2:
            critic, _ = disc([real_image, y]) if use_y else disc(real_image)

        elif output == 3:
            critic, _, _ = disc([real_image, y]) if use_y else disc(real_image)

        elif output == 4:
            critic, _, _, _ = disc([real_image, y]) if use_y else disc(real_image)

        d = tf.reduce_sum(critic)
    grads = tape.gradient(d, real_image)
    grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    r1 = tf.reduce_mean(tf.square(grad_norms))
    return r1


def adv_loss(critic_real, critic_fake, real_image=None, fake_image=None, disc=None, gan_loss='gan',
             gp_type='none', output=1, y=None):
    # gan objective
    if gan_loss == 'gan':
        d_loss = tf.reduce_mean(
            bce(tf.ones_like(critic_real), critic_real) + \
            bce(tf.zeros_like(critic_fake), critic_fake))
        g_loss = tf.reduce_mean(
            bce(tf.ones_like(critic_fake), critic_fake))

    elif gan_loss == 'wgan':
        d_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)
        g_loss = -tf.reduce_mean(critic_fake)

    elif gan_loss == 'nsgan':
        d_loss = tf.reduce_mean(tf.math.softplus(critic_fake)) + \
                 tf.reduce_mean(tf.math.softplus(-critic_real))
        g_loss = tf.reduce_mean(tf.math.softplus(-critic_fake))

    elif gan_loss == 'lsgan':
        d_loss = 0.5 * tf.reduce_mean(
            (critic_real - 1) ** 2 + (critic_fake) ** 2)
        g_loss = 0.5 * tf.reduce_mean((critic_fake - 1) ** 2)

    # gradient penalty regularzation
    if gp_type == 'none':
        gp = 0.

    elif gp_type == 'wgan':
        gp = gradient_penalty(real_image, fake_image, disc, y)

    elif gp_type == 'r1':
        gp = r1_regularzation(real_image, disc, output, y)

    return d_loss, g_loss, gp





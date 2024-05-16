import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, activations, constraints, initializers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import backend as K


class NoiseInjection(layers.Layer):
    def __init__(self):
        super().__init__()
        self.scale = self.add_weight('scale', shape=[1], initializer=initializers.Constant(1e-2))

    def call(self, x):
        b, h, w, c = x.shape
        noise = tf.random.normal((b, h, w, 1))
        return x + self.scale * noise


class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class l2_normalization(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, shape):
        dim = shape[-1]
        self.alpha = self.add_weight(shape=[1, 1, dim], trainable=True, initializer='ones', name='alpha')
        self.beta = self.add_weight(shape=[1, 1, dim], trainable=True, initializer='zeros', name='beta')

    def call(self, x):
        return self.alpha * tf.math.l2_normalize(x, axis=-1) + self.beta


class LinearBlock(layers.Layer):
    def __init__(self, filters, norm='none', act='relu'):
        super().__init__()

        self.fc = layers.Dense(filters, kernel_initializer='he_normal')

        # normalization
        if norm == 'bn':
            self.norm = layers.BatchNormalization()

        elif norm == 'in':
            self.norm = tfa.layers.Instance

        elif norm == 'l2':
            self.norm = l2_normalization()

        elif norm == 'none':
            self.norm = layers.Lambda(lambda x: x)

        # activation
        if act == 'relu':
            self.act = layers.ReLU()

        elif act == 'lrelu':
            self.act = tf.keras.Sequential([
                layers.LeakyReLU(0.2),
                layers.Lambda(lambda x: x / tf.sqrt(2.))])

        elif act == 'tanh':
            self.act = activations.tanh

        elif act == 'none':
            self.act = layers.Lambda(lambda x: x)

    def call(self, x, training=False):
        x = self.fc(x)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x


class AdaptiveInstanceNormalization(layers.Layer):
    def __init__(self):
        super().__init__()
        self.norm = tfa.layers.InstanceNormalization()

    def build(self, shape):
        x_shape, w_shape = shape
        self.dim = x_shape[-1]
        self.scaleMap = layers.Dense(self.dim)
        self.offsetMap = layers.Dense(self.dim)

    def call(self, inputs):
        x, w = inputs
        gamma = tf.reshape(self.scaleMap(w), [-1, 1, 1, self.dim])
        beta = tf.reshape(self.offsetMap(w), [-1, 1, 1, self.dim])
        x = gamma * self.norm(x) + beta
        return x


class Conv2DMod(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 kernel_initializer='he_normal',
                 demod=True,
                 **kwargs):
        super(Conv2DMod, self).__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = padding
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.demod = demod
        self.input_spec = [layers.InputSpec(ndim=4),
                           layers.InputSpec(ndim=2)]

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel')

        # Set input spec.
        self.input_spec = [layers.InputSpec(ndim=4, axes={channel_axis: input_dim}),
                           layers.InputSpec(ndim=2)]
        self.built = True

    def call(self, inputs):
        x, w = inputs

        # w:(b,dim)

        # (b,1,1,dim,1)
        inp_mods = K.expand_dims(K.expand_dims(K.expand_dims(w, axis=1), axis=1), axis=-1)

        # kernel:(1,3,3,in,out)
        my_kernel = K.expand_dims(self.kernel, axis=0)

        weights = my_kernel * (inp_mods + 1)

        # Demodulate
        if self.demod:
            # Get variance by each output channel
            d = K.sqrt(K.sum(K.square(weights), axis=[1, 2, 3], keepdims=True) + 1e-8)
            weights = weights / d

        # Fuse kernels and fuse inputs
        x = tf.transpose(x, [0, 3, 1, 2])  # [BHWC] -> [BCHW]
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])  # [1, if*bs, h, w]

        # Kernel should be 3x3, from inp_fil*bs, out_fil

        w = tf.transpose(weights, [1, 2, 3, 0, 4])  # [3, 3, input_maps, bs, output_maps]
        w = tf.reshape(w, [weights.shape[1], weights.shape[2], weights.shape[3],
                           -1])  # [3, 3, input_maps, output_maps*batch_size]

        x = tf.nn.conv2d(x, w,
                         strides=self.strides,
                         padding='SAME' if self.padding == 'same' else 'VALID',
                         data_format="NCHW")

        # print(x.shape)

        # Un-fuse output
        x = tf.reshape(x, [-1, self.filters, tf.shape(x)[2],
                           tf.shape(x)[3]])  # Fused => reshape convolution groups back to minibatch.
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


class ConvBlock(layers.Layer):
    def __init__(self,
                 filters,
                 ks,
                 st,
                 padding='same',
                 norm='none',
                 act='relu',
                 use_bias=True,
                 act_first=False,
                 transpose=False,
                 reflect=False,
                 ):
        super().__init__()

        self.act_first = act_first

        # normalization
        if norm == 'bn':
            self.norm = layers.BatchNormalization()

        elif norm == 'in':
            self.norm = tfa.layers.InstanceNormalization()

        elif norm == 'l2':
            self.norm = l2_normalization()

        elif norm == 'adain':
            self.norm = 'adain'

        elif norm == 'none':
            self.norm = layers.Lambda(lambda x: x)

        # activation
        if act == 'relu':
            self.act = layers.ReLU()

        elif act == 'lrelu':
            self.act = tf.keras.Sequential([
                layers.LeakyReLU(0.2),
                layers.Lambda(lambda x: x / tf.sqrt(2.))])

        elif act == 'tanh':
            self.act = activations.tanh

        elif act == 'sigmoid':
            self.act = tf.nn.sigmoid

        elif act == 'none':
            self.act = layers.Lambda(lambda x: x)

        # conv
        if transpose:
            self.conv = layers.Conv2DTranspose(filters,
                                               ks,
                                               st,
                                               padding=padding,
                                               use_bias=use_bias,
                                               kernel_initializer='he_normal'
                                               )
        elif norm == 'adain':
            self.conv = Conv2DMod(filters, ks, st, padding)
        else:
            self.conv = layers.Conv2D(
                filters,
                ks,
                st,
                padding='valid' if reflect else padding,
                use_bias=use_bias,
                kernel_initializer='he_normal'
            )

        if reflect:
            self.pad = ReflectionPadding2D()
        else:
            self.pad = layers.Lambda(lambda x: x)

    def build(self, shape):
        if self.norm == 'adain':
            dim = shape[0][-1]
            self.wMap = layers.Dense(dim)
            self.noise = NoiseInjection()

    def call(self, inputs, training=False):
        if self.norm == 'adain':
            x, w = inputs
            w = self.wMap(w)
        else:
            x = inputs

        if self.act_first:
            x = self.act(x)
            if self.norm == 'adain':
                x = self.noise(self.conv([x, w]))
            else:
                x = self.conv(self.pad(x))
                x = self.norm(x, training=training)
        else:
            if self.norm == 'adain':
                x = self.noise(self.conv([x, w]))
            else:
                x = self.conv(self.pad(x))
                x = self.norm(x, training=training)
            x = self.act(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self, filters,
                 norm='in',
                 act='relu',
                 act_first=False,
                 downsample=False,
                 upsample=False,
                 reflect=False):
        super().__init__()

        self.conv1 = ConvBlock(filters, 3, 1, norm=norm, act=act, act_first=act_first,
                               reflect=True if reflect else False)

        self.conv2 = ConvBlock(filters,
                               3,
                               1,
                               norm=norm,
                               act='none' if not act_first else act,
                               act_first=act_first,
                               reflect=True if reflect else False
                               )
        self.norm = norm
        self.filters = filters

        self.downsample = downsample
        self.upsample = upsample

        self.up = layers.UpSampling2D(interpolation='bilinear')
        self.down = layers.AveragePooling2D()

    def build(self, shape):
        if self.norm == 'adain':
            x_shape, w_shape = shape
        else:
            x_shape = shape

        dim = x_shape[-1]

        if dim != self.filters:
            self.skip = ConvBlock(self.filters,
                                  1,
                                  1,
                                  padding='valid',
                                  norm='none',
                                  act='none',
                                  use_bias=False)
        else:
            self.skip = layers.Lambda(lambda x: x)

    def call(self, inputs, training=False):
        if self.norm == 'adain':
            x, w = inputs
        else:
            x = inputs
        skip = x
        if self.upsample:
            x = self.up(x)
            skip = x
        if self.downsample:
            skip = self.down(x)
        skip = self.skip(skip)
        if self.norm == 'adain':
            x = self.conv1([x, w], training=training)
            if self.downsample:
                x = self.down(x)
            x = self.conv2([x, w], training=training)
        else:
            x = self.conv1(x, training=training)
            if self.downsample:
                x = self.down(x)
            x = self.conv2(x, training=training)
        x = (x + skip) / tf.sqrt(2.)
        return x
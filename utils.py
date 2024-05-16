import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from models import SimaGAN, SimaGAN_AC, YVAE, StarGAN, ICGAN, AttGAN, ELEGANT


def load_model(opt):
    config = get_config(f'./configs/{opt.model}.yaml')
    if opt.model == 'SimaGAN':
        model = SimaGAN.SimaGAN(config)
        key_var = f"{config['tau']}_{config['lambda_r']}_{config['lambda_dce']}"

    elif opt.model == 'SimaGAN_AC':
        model = SimaGAN_AC.SimaGAN_AC(config)
        key_var = f"{config['lambda_r']}"

    elif opt.model == 'YVAE':
        model = YVAE.YVAE(config)
        key_var = f"{config['lambda_ce']}_{config['lambda_zr']}_{config['beta']}"

    elif opt.model == 'StarGAN':
        model = StarGAN.StarGAN(config)
        key_var = f"{config['lambda_r']}"

    elif opt.model == 'ICGAN':
        model = ICGAN.ICGAN(config)
        key_var = "none"

    elif opt.model == 'AttGAN':
        model = AttGAN.AttGAN(config)
        key_var = f"{config['lambda_r']}_{config['lambda_cls_g']}_{config['lambda_cls_d']}"

    elif opt.model == 'ELEGANT':
        model = ELEGANT.ELEGANT(config)
        key_var = f"{config['norm']}"
    return model, key_var, config


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_image(pth, label, image_size):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=3)
    image = tf.cast(tf.image.resize(image, (image_size, image_size)), 'float32')
    return image / 255., label


def build_df(pth, val_ratio):
    train_folder = sorted(os.listdir(pth))
    class_map = {name: i for i, name in enumerate(train_folder)}
    pths = []
    labels = []

    for c in train_folder:
        for name in sorted(os.listdir(pth + '/' + c)):
            pths.append(pth + '/' + c + '/' + name)
            labels.append(class_map[c])
    df = pd.DataFrame(np.vstack([pths, labels]).T, columns=['pth', 'label'])
    df['label'] = df.label.map(lambda x: int(x))

    if val_ratio is not None:
        df_train, df_val = train_test_split(df, test_size=val_ratio, random_state=999)
    else:
        df_val = None
    return df_train, df_val, class_map


def build_ds(pth, image_size, batch_size, val_ratio=None):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    df_train, df_val, class_map = build_df(pth, val_ratio)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (df_train['pth'], df_train['label'])).shuffle(1024). \
        map(lambda path, label: get_image(path, label, image_size)). \
        batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

    if df_val is not None:
        ds_val = tf.data.Dataset.from_tensor_slices(
            (df_val['pth'], df_val['label'])). \
            map(lambda path, label: get_image(path, label, image_size), num_parallel_calls=AUTOTUNE). \
            batch(64, drop_remainder=True).prefetch(AUTOTUNE)
    else:
        ds_val = None
    return ds_train, ds_val, class_map


def ref_sample(source, ref, G, opt, epoch, key_var, y1=None, y2=None):
    b, h, w, c = ref.shape
    xa_repeat = tf.repeat(source, b, axis=0)
    xb_repeat = tf.reshape(tf.stack([ref for _ in range(b)], axis=0), (b ** 2, h, w, c))

    ca, sa = G.encode(xa_repeat)
    cb, sb = G.encode(xb_repeat)

    if opt.model == 'SimaGAN' or opt.model == 'SimaGAN_AC':
        xa2b = G.decode(ca, sb)
    elif opt.model == 'ELEGANT':
        y1 = tf.repeat(y1, b, axis=0)
        y2 = tf.reshape(tf.stack([y2 for _ in range(b)], axis=0), (b ** 2,))
        xa2b = G.decode(xa_repeat, ca, cb, y1, y2, sa)
    fig, ax = plt.subplots(ncols=b + 1, nrows=b + 1, figsize=(8, 8))

    for k in range(b + 1):
        if k == 0:
            ax[0, k].imshow(tf.ones(source[0].shape))
            ax[0, k].axis('off')
        else:
            ax[0, k].imshow(source[k - 1])
            ax[0, k].axis('off')

    for k in range(1, b + 1):
        ax[k, 0].imshow(ref[k - 1])
        ax[k, 0].axis('off')

    k = 0
    for j in range(b):
        for i in range(b):
            ax[i + 1, j + 1].imshow(xa2b[k])
            ax[i + 1, j + 1].axis('off')
            k += 1
    plt.tight_layout()

    pth = f'{opt.output_dir}/{opt.model}/image_{key_var}'
    if not os.path.exists(pth):
        os.makedirs(pth)
    plt.savefig(f'{pth}/synthesis_{epoch}.jpg')


def label_sample(x, G, opt, epoch, class_map, key_var):
    x = x[:1]
    try:
        nclass = G.D.c_dim
    except:
        nclass = G.nclass
    fig, ax = plt.subplots(ncols=nclass + 1)
    ax[0].imshow(x[0])
    ax[0].axis('off')
    ax[0].set_title('x')

    try:
        factors, _ = G.encode(x)
        z = G.reparameterize(factors, 0.)
        for i in range(nclass):
            y = tf.constant([i])
            x_t = G.decode(y, z)
            ax[i + 1].imshow(x_t[0])
            ax[i + 1].axis('off')
            ax[i + 1].set_title(f'{class_map[i]}')
    except:
        for i in range(nclass):
            y = tf.constant([i])
            if opt.model != 'ICGAN':
                try:
                    x_t, _, _ = G([x, y])
                except:
                    x_t = G([x, y])
            else:
                z, _ = G.encode(x)
                x_t = G.decode(z, y)
            ax[i + 1].imshow(x_t[0])
            ax[i + 1].axis('off')
            ax[i + 1].set_title(f'{class_map[i]}')

    plt.tight_layout()
    pth = f'{opt.output_dir}/{opt.model}/image_{key_var}'
    if not os.path.exists(pth):
        os.makedirs(pth)
    plt.savefig(f'{pth}/synthesis_{epoch}.jpg')


class VisualizeRefCallback(callbacks.Callback):
    def __init__(self, source, ref, opt, class_map, key_var, y_source=None, y_ref=None):
        super().__init__()
        self.source = source
        self.ref = ref
        self.opt = opt
        self.key_var = key_var
        self.class_map = {v: k for k, v in class_map.items()}
        self.y_source = y_source
        self.y_ref = y_ref

    def on_epoch_end(self, epoch, logs=None):
        if self.opt.model == 'SimaGAN' or self.opt.model == 'ELEGANT' or self.opt.model == 'SimaGAN_AC':
            ref_sample(self.source, self.ref, self.model.G, self.opt, epoch, self.key_var, self.y_source, self.y_ref)
        else:
            label_sample(self.source, self.model.G, self.opt, epoch, self.class_map, self.key_var)


def set_callbacks(config, opt, sample, key_var, class_map, y):
    # ckpt
    pth = f"{opt.ckpt_dir}/{opt.model}"
    if not os.path.exists(pth):
        os.makedirs(pth)
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"{pth}/{opt.model}_{key_var}/{opt.model}",
        save_weights_only=True)

    # results
    pth = f"{opt.output_dir}/{opt.model}"
    if not os.path.exists(pth):
        os.makedirs(pth)
    history_callback = callbacks.CSVLogger(
        f"{pth}/log_{opt.model}_{key_var}.csv",
        separator=",",
        append=False)

    # visualize callbacks
    visualize_callback = VisualizeRefCallback(sample, sample, opt, class_map, key_var, y, y)
    callbacks_ = [model_checkpoint_callback, history_callback, visualize_callback]
    return callbacks_
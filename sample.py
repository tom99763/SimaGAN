import argparse
import os
import matplotlib.pyplot as plt
from utils import *
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SimaGAN')
    parser.add_argument('--img_dir', type=str, default='./examples')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./example_results')
    parser.add_argument('--image_size', type=int, default=128)
    opt, _ = parser.parse_known_args()
    return opt

def load_image(pth, image_size):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=3)
    image = tf.cast(tf.image.resize(image, (image_size, image_size)), 'float32')
    return image / 255.

def build_ds(opt):
    fer = sorted(os.listdir(opt.img_dir))
    fer_imgs = []
    for cls_dir in fer:
        cls_dir = f'{opt.img_dir}/{cls_dir}'
        img_dir = list(map(lambda x:f'{cls_dir}/{x}', os.listdir(cls_dir)))[3]
        fer_imgs.append(img_dir)
    ds = tf.data.Dataset.from_tensor_slices(fer_imgs). \
        map(lambda path: load_image(path, opt.image_size)). \
        batch(5).prefetch(AUTOTUNE)
    return ds


def interpolate(x, params, model, opt):
    xa, xb = tf.split(x, 2, axis=0)
    ca, fa = model.G.encode(xa)
    cb, fb = model.G.encode(xb)
    fig, ax = plt.subplots(ncols=8, nrows=4, figsize=(16, 16))
    for i, xi in enumerate(xa):
        ax[i, 0].imshow(xi)
        ax[i, 0].axis('off')
        eps = (fb[i] - fa[i]) / 6
        for j in range(1, 7):
            f = fa[i] + eps * j
            f = f[None, ...]
            xt = model.G.decode(ca[i][None, ...], f)
            ax[i, j].imshow(xt[0])
            ax[i, j].axis('off')
        ax[i, 7].imshow(xb[i])
        ax[i, 7].axis('off')
    dir = f'{opt.output_dir}/{opt.model}_{params}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/interpolate.png')


def reference(x, params, model, opt):
    src, ref = x, x
    b, h, w, c = ref.shape
    xa_repeat = tf.repeat(src, b, axis=0)
    xb_repeat = tf.reshape(tf.stack([ref for _ in range(b)], axis=0), (b ** 2, h, w, c))
    ca, fa = model.G.encode(xa_repeat)
    cb, fb = model.G.encode(xb_repeat)
    xab = model.G.decode(ca, fb)
    fig, ax = plt.subplots(ncols=b + 1, nrows=b + 1, figsize=(8, 8))
    for k in range(b + 1):
        if k == 0:
            ax[0, k].imshow(tf.ones(src[0].shape))
            ax[0, k].axis('off')
        else:
            ax[0, k].imshow(src[k - 1])
            ax[0, k].axis('off')

    for k in range(1, b + 1):
        ax[k, 0].imshow(ref[k - 1])
        ax[k, 0].axis('off')

    k = 0
    for j in range(b):
        for i in range(b):
            ax[i + 1, j + 1].imshow(xab[k])
            ax[i + 1, j + 1].axis('off')
            k += 1
    plt.tight_layout()
    dir = f'{opt.output_dir}/{opt.model}_{params}'
    plt.savefig(f'{dir}/reference.png')


def main():
    opt = parse_opt()
    #load model
    model, key_var, config = load_model(opt)
    # load model weights
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}/{opt.model}_{key_var}"
    model.load_weights(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

    #dataset
    ds = build_ds(opt)

    #viz
    for x in ds:
        #interpolate(x, key_var, model, opt)
        reference(x, key_var, model, opt)

if __name__ == '__main__':
    main()
import argparse
import os
import matplotlib.pyplot as plt
from utils import *
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AttGAN')
    parser.add_argument('--img_dir', type=str, default='./examples')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./example_results')
    parser.add_argument('--image_size', type=int, default=128)
    opt, _ = parser.parse_known_args()
    return opt


def reference(x, params, model, opt, y1=None, y2=None):
    src, ref = x, x
    b, h, w, c = ref.shape
    xa_repeat = tf.repeat(src, b, axis=0)
    xb_repeat = tf.reshape(tf.stack([ref for _ in range(b)], axis=0), (b ** 2, h, w, c))
    ca, fa = model.G.encode(xa_repeat)
    cb, fb = model.G.encode(xb_repeat)

    if opt.model == 'SimaGAN' or opt.model == 'SimaGAN_AC':
        xab = model.G.decode(ca, fb)
    elif opt.model == 'ELEGANT':
        y1 = tf.repeat(y1, b, axis=0)
        y2 = tf.reshape(tf.stack([y2 for _ in range(b)], axis=0), (b ** 2,))
        xab = model.G.decode(xa_repeat, ca, cb, y1, y2, fa)

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
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/result.png')


def label_sample(x, y, model, opt, params):
    b, h, w, c = x.shape

    fig, ax = plt.subplots(ncols=b + 1, nrows =  b)
    for i in range(b):
        x_ = x[i][None,...]
        ax[i,0].imshow(x_[0])
        ax[i,0].axis('off')

        x_ = tf.repeat(x_, b, axis=0)

        if opt.model != 'ICGAN':
            try:
                x_t, _, _ = model.G([x_, y])
            except:
                x_t = model.G([x_, y])
        else:
            z, _ = model.G.encode(x)
            x_t = model.G.decode(z, y)
        for j in range(1, b+1):
            ax[i, j].imshow(x_t[j-1])
            ax[i, j].axis('off')

    plt.tight_layout()
    dir = f'{opt.output_dir}/{opt.model}_{params}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f'{dir}/result.png')


def load_image(pth, image_size, label):
    image = tf.image.decode_jpeg(tf.io.read_file(pth), channels=3)
    image = tf.cast(tf.image.resize(image, (image_size, image_size)), 'float32')
    return image / 255., tf.cast(label, 'int64')

def build_ds(opt):
    fer = sorted(os.listdir(opt.img_dir))
    fer_imgs = []
    labels = []
    for i, cls_dir in enumerate(fer):
        cls_dir = f'{opt.img_dir}/{cls_dir}'
        img_dir = list(map(lambda x:f'{cls_dir}/{x}', os.listdir(cls_dir)))[3]
        fer_imgs.append(img_dir)
        labels.append(i)
    ds = tf.data.Dataset.from_tensor_slices((fer_imgs, labels)). \
        map(lambda path, label: load_image(path, opt.image_size, label)). \
        batch(4).prefetch(AUTOTUNE)
    return ds

def main():
    opt = parse_opt()
    # load model
    model, params, config = load_model(opt)
    # load model weights
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}/{opt.model}_{params}"
    model.load_weights(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    ds = build_ds(opt)
    print(params)
    for x, y in ds:
        if opt.model == 'SimaGAN':
            reference(x, params, model, opt)
        elif opt.model == 'ELEGANT':
            reference(x, params, model, opt, y, y)
        else:
            label_sample(x, y, model, opt, params)

if __name__ == '__main__':
    main()
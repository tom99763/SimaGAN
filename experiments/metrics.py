from tensorflow.keras import callbacks, layers, optimizers, losses
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.stats import entropy
from scipy.linalg import sqrtm
import tensorflow as tf
import utils
import argparse
import numpy as np
import os
from tqdm import tqdm


def psnr_score(x1, x2):
    return tf.image.psnr(x1, x2, max_val=1.0)


def ssim_score(x1, x2):
    return tf.image.ssim(x1, x2, max_val=1.0)


def calculate_fid(r_e, f_e):
    # calculate mean and covariance statistics
    mu1, sigma1 = r_e.mean(axis=0), np.cov(r_e, rowvar=False)
    mu2, sigma2 = f_e.mean(axis=0), np.cov(f_e, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def psnr_score(x1, x2):
    return tf.image.psnr(x1, x2, max_val=1.0)


def ssim_score(x1, x2):
    return tf.image.ssim(x1, x2, max_val=1.0)


def calculate_fid(r_e, f_e):
    # calculate mean and covariance statistics
    mu1, sigma1 = r_e.mean(axis=0), np.cov(r_e, rowvar=False)
    mu2, sigma2 = f_e.mean(axis=0), np.cov(f_e, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

class Scorer(tf.keras.Model):
    def __init__(self, opt, num_classes=8):
        super().__init__()
        self.opt = opt
        self.inception_model = InceptionV3(include_top=False,
                                           weights="imagenet",
                                           pooling='avg')
        self.inception_model.trainable = False
        self.phead = layers.Dense(num_classes)
        self.num_classes = num_classes

    def call(self, x, output_emb=False):
        x = self.preprocess(x)
        emb = self.inception_model(x)
        logits = self.phead(emb)

        if output_emb:
            return tf.nn.softmax(logits, axis=-1), emb
        else:
            return logits

    def preprocess(self, x):
        x *= 255.
        x = tf.image.resize(x, (299, 299))
        x = preprocess_input(x)
        return x

    def compute_score(self, model, ds_content, ds_style):
        '''
        ds_content: a image set with the same domain
        x_style, y_style: a image set with another domain
        '''
        # metrics record
        all_preds = []
        IS = []
        CIS = []
        PSNR = []
        SSIM = []
        R_E = []
        F_E = []

        # time complexity: O(nm)
        for i, (x_content, y_content) in tqdm(enumerate(ds_content)):
            # record the preds of inceptionV3
            cur_preds = []
#61
            # iter over each style feature
            for x_style, y_style in ds_style:
                x_style = x_style[None, ...]
                # extend shape
                y_style = tf.repeat(y_style, x_content.shape[0], axis=0)

                if self.opt.model == 'SimaGAN':
                    # content feature
                    c, _ = model.encode(x_content)
                    _, sj = model.encode(x_style)
                    sj = tf.repeat(sj, x_content.shape[0], axis=0)

                    # stylize content image
                    x_styled = model.decode(c, sj)

                elif self.opt.model == 'ELEGANT':
                    # content feature
                    zc, fmaps = model.encode(x_content)
                    zs, _ = model.encode(x_style)
                    zs = tf.repeat(zs, x_content.shape[0], axis=0)

                    # styleize content image
                    x_styled = model.decode(x_content, zc, zs, y_content, y_style, fmaps)

                elif self.opt.model == 'StarGAN':
                    x_styled = model([x_content, y_style])

                elif self.opt.model == 'ICGAN':
                    z, _ = model.encode(x_content)
                    x_styled = model.decode(z, y_style)

                elif self.opt.model == 'AttGAN':
                    _, z, fmaps = model([x_content, y_content])
                    x_styled = model.decode(z, y_style, fmaps)

                prob, f_e = self.call(x_styled, output_emb=True)

                # predict
                if i == 0:
                    _, r_e = self.call(x_style, output_emb=True)
                    R_E.append(r_e)

                ##record
                # compute quality metrics
                PSNR.append(psnr_score(x_content, x_styled))
                SSIM.append(ssim_score(x_content, x_styled))
                all_preds.append(prob)  # used for compute IS
                cur_preds.append(prob)  # used for compute CIS
                F_E.append(f_e)

            # compute CIS
            cur_preds = tf.concat(cur_preds, axis=0)
            py = tf.math.reduce_sum(cur_preds, axis=0)
            for j in range(cur_preds.shape[0]):
                pyx = cur_preds[j, :]
                CIS.append(entropy(pyx, py))

        # compute IS
        all_preds = tf.concat(all_preds, axis=0)
        py = tf.math.reduce_sum(all_preds, axis=0)
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

        # compute metrics
        IS = tf.exp(tf.reduce_mean(IS))
        CIS = tf.exp(tf.reduce_mean(CIS))
        R_E = tf.concat(R_E, axis=0)
        F_E = tf.concat(F_E, axis=0)
        FID = calculate_fid(R_E.numpy(), F_E.numpy())
        PSNR = tf.reduce_mean(PSNR)
        SSIM = tf.reduce_mean(SSIM)
        return IS, CIS, FID, PSNR, SSIM



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../datasets/fer/RAFD',
                        help='directory of training dataset')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='directory to store checkpoints')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--val_ratio', type=int, default=0.2, help='validation split ratio')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    opt, _ = parser.parse_known_args()
    return opt


def main(opt):
    scorer = Scorer()
    scorer.compile(
        optimizer=optimizers.Adam(learning_rate=opt.lr),
        metrics='acc',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # set callbacks
    pth = f"{opt.ckpt_dir}/Scorer"
    if not os.path.exists(pth):
        os.makedirs(pth)
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=f"{pth}/Scorer",
        save_weights_only=True)

    # build dataset
    ds_train, ds_val, class_map = utils.build_ds(opt.train_dir, opt.image_size, opt.batch_size, opt.val_ratio)

    # train
    scorer.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=opt.num_epochs,
        callbacks=[checkpoint_callback]
    )


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
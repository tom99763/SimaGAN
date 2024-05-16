from experiments.metrics import Scorer
from utils import *
import os
from sklearn.model_selection import train_test_split as ttp
import argparse

AUTOTUNE = tf.data.experimental.AUTOTUNE

def create_evaluate_dataset(opt):
    _, df_val, _ = build_df(opt.train_dir, opt.val_ratio)
    df_content, df_style = ttp(df_val, test_size=opt.test_size, shuffle=False, random_state=999)
    ds_content = tf.data.Dataset.from_tensor_slices(
        (df_content['pth'], df_content['label'])). \
        map(lambda path, label: get_image(path, label, opt.image_size)). \
        batch(opt.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    ds_style = tf.data.Dataset.from_tensor_slices(
        (df_style['pth'], df_style['label'])). \
        map(lambda path, label: get_image(path, label, opt.image_size)). \
        prefetch(AUTOTUNE)
    return ds_content, ds_style


def main(opt):
    # dataset
    ds_content, ds_style = create_evaluate_dataset(opt)

    # load scorer weights
    scorer = Scorer(opt)
    scorer.load_weights(tf.train.latest_checkpoint(f'{opt.ckpt_dir}/Scorer'))

    # load MODEL
    model, key_var, config = load_model(opt)

    # load model weights
    ckpt_dir = f"{opt.ckpt_dir}/{opt.model}/{opt.model}_{key_var}"
    model.load_weights(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

    # evaluate score
    IS, CIS, FID, PSNR, SSIM = scorer.compute_score(model.G, ds_content, ds_style)

    print(f'model: {opt.model} -- IS: {IS} -- CIS: {CIS} -- FID: {FID} -- PSNR: {PSNR} -- SSIM: {SSIM}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SimaGAN', help='which model used to train')
    parser.add_argument('--train_dir', type=str, default='../datasets/faces/fer/RAFD',
                        help='directory of training dataset')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='directory to store checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--val_ratio', type=int, default=0.2, help='validation split ratio')
    parser.add_argument('--test_size', type=int, default=0.1)
    opt, _ = parser.parse_known_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

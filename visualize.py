import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from utils import *
from tqdm import tqdm


def main(opt):
    ds_train, ds_val, class_map = build_ds(opt.train_dir, opt.image_size, opt.batch_size, val_ratio=opt.val_ratio)
    class_map = {v: k for k, v in class_map.items()}
    model, key_var, config = load_model(opt)
    tsne = TSNE(perplexity=300.0, init='pca', random_state=999, n_iter=2000)
    for name in os.listdir(opt.ckpt_dir + '/' + 'SimaGAN'):
        tau = float(name.split('_')[1])

        # load model weights
        ckpt_dir = opt.ckpt_dir + '/' + 'SimaGAN' + '/' + name
        model.load_weights(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

        features = []
        labels = []
        for x, y in tqdm(ds_train):
            c, s = model.G.encode(x)
            features.append(s)
            labels.append(y)

        features = tf.concat(features, axis=0).numpy()
        labels = tf.concat(labels, axis=0).numpy()
        manifolds = tsne.fit_transform(features)

        plt.figure()
        plt.title(f'$\\tau$ = {tau}', fontsize=20)

        for i in range(8):
            x = manifolds[labels == i]
            plt.scatter(x[:, 0], x[:, 1], label=class_map[i], s=10, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(f'{opt.result_dir}/{opt.model}/{tau}.png', bbox_inches='tight')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SimaGAN', help='which model used to train')
    parser.add_argument('--train_dir', type=str, default='../datasets/fer/RAFD',
                        help='directory of training dataset')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='directory to store checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--val_ratio', type=int, default=0.2, help='validation split ratio')
    opt, _ = parser.parse_known_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

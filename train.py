import argparse
from utils import *
from tensorflow.keras import optimizers, callbacks
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../datasets/fer/RAFD', help='directory of training dataset')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='directory to store checkpoints')
    parser.add_argument('--output_dir', type=str, default='./results', help='directory to store results')
    parser.add_argument('--model', type=str, default='SimaGAN', help='which model used to train')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--val_ratio', type=int, default=0.2, help='validation split ratio')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.0, help='momentum of adam')
    parser.add_argument('--beta_2', type=float, default=0.99, help='momentum of adam')
    opt, _ = parser.parse_known_args()
    return opt


def main(opt):
    if opt.gpus == 1:
        # select model
        model, key_var, config = load_model(opt)

        # build dataset
        ds_train, ds_val, class_map = build_ds(opt.train_dir, opt.image_size, opt.batch_size, opt.val_ratio)

        # set optimizer
        model.compile(optimizer=[
            optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
            optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2)])

    else:
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=communication_options)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        options.experimental_distribute.num_devices = opt.gpus

        with strategy.scope():
            model, key_var, config = load_model(opt)

            # build dataset
            ds_train, ds_val, class_map = build_ds(opt.train_dir, opt.image_size, opt.batch_size, opt.val_ratio)
            ds_train = ds_train.with_options(options)
            ds_val = ds_val.with_options(options)
            # set optimizer
            model.compile(optimizer=[
                optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
                optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2)])

    ckpt = f"{opt.ckpt_dir}/{opt.model}/{opt.model}_{key_var}"
    if os.path.exists(ckpt):
        model.load_weights(tf.train.latest_checkpoint(ckpt))

    # set callbacks
    sample, y = next(iter(ds_val.take(1)))
    callbacks_ = set_callbacks(config, opt, sample[:5], key_var, class_map, y[:5])

    # train
    model.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=opt.num_epochs,
        callbacks=callbacks_
    )


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
import tsgm
import argparse
import keras
import tensorflow as tf
import math

from utils import to_tensor_dataset, load_config, load_numpy_data

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--ckpt',type=str)
parser.add_argument('--config',type=str)
parser.add_argument('--vis',type=str)

args = parser.parse_args()

config = load_config(args.config)

(train_data,train_label) , (_,_) = load_numpy_data(args.data)

dataset = to_tensor_dataset(train_data,train_label,batch_size=config['batch_size'])

architecture = tsgm.models.architectures.zoo(config['architecture'])(
    seq_len = config['seq_len'], feat_dim = config['feature_dim'],
    latent_dim = config['latent_dim'], output_dim = config['output_dim']
)

discriminator, generator = architecture.discriminator, architecture.generator

cond_gan = tsgm.models.cgan.ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=config['latent_dim']
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

n_batches = len(train_data) / config['batch_size']
n_batches = math.ceil(n_batches)
checkpoint_path = args.ckpt + config['architecture'] +"/ckpt-{epoch:04d}.ckpt"


cbk = tsgm.models.monitors.GANMonitor(num_samples=3, latent_dim=config['latent_dim'], save=True, labels=train_label, save_path=args.vis)
cbk2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_freq=config['save_freq'] * n_batches)
cond_gan.fit(dataset, epochs=config['epochs'], callbacks=[cbk,cbk2])




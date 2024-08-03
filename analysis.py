import torch
import numpy as np
from pathlib import Path
import os

# from models.generative.gan import eeg_cgan
from models.generative.diffusion.diffusion_ts import Diffusion
from models.generative.diffusion.unet1d import Unet1D
from models.generative.diffusion.transformer import Transformer

from utils import load_config
from model_utils import  get_trainer_from_config
from utils import load_numpy_data
from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample


from argument import analysis_argument

def main():
    args = analysis_argument()
    config = load_config(args.config)
    # (train_data,train_label) , (test_data,test_label) = load_numpy_data(args.data)

    train_data = np.load(args.data,allow_pickle=True).item()
    train_data = np.array(train_data[args.task])
    print(train_data.shape)

    trainer = get_trainer_from_config(args,config,curr_date=args.curr_date)
    trainer.load_weight(os.path.join(args.ckpt,args.curr_date,"checkpoint.pth"))
    samples  = trainer.generate_samples(num_samples=100,num_per_batch=10)
    print(samples.shape)

    save_path = os.path.join(args.save,Path(args.ckpt).stem, args.task)
    os.makedirs(save_path,exist_ok=True)

    np.save(os.path.join(save_path,'synthesize.npy'), samples)

    plot_pca(real=train_data,fake=samples,save_path=save_path)
    plot_tsne(real=train_data,fake=samples,save_path=save_path)
    plot_umap(real=train_data,fake=samples,save_path=save_path)
    plot_sample(real=train_data,fake=samples,save_path=save_path)


if __name__ == '__main__':
    main()













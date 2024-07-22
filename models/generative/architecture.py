import PIL.Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import PIL
from torchvision.transforms import ToTensor
import math

from einops import reduce
from tqdm.auto import tqdm
from functools import partial

from train_utils import gradient_panelty

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def exists(x):
    return x is not None

def default(val,d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t,*args,**kwargs):
    return t

def extract(a,t,x_shape):
    b, *_ = t.shape
    out = a.gather(-1,t)
    return out.reshape(b,*((1,) * (len(x_shape) - 1)))


class ConditionalGAN:
    def __init__(self,generator,discriminator,
                 g_optimizer,d_optimizer,
                 g_scheduler,d_scheduler,
                 criterion,lambda_cls,lambda_gp,
                 max_iter,save_iter,n_critic,num_classes,latent_dim,
                 writer,save_path):
        
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler
        self.criterion = criterion
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.lambda_gp = lambda_gp
        self.lamdba_cls = lambda_cls
        self.max_iter = max_iter
        self.save_iter = save_iter
        self.n_critic = n_critic
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.writer = writer
        self.save_path = save_path

    def train(self,dataloader):
        '''Train Coniditional GAN model'''
        dataloader_cycle = cycle(dataloader)

        self.generator.train()
        self.discriminator.train()

        for iter in tqdm(range(self.max_iter)):
            sequence , label = next(dataloader_cycle)
            sequence = sequence.to(self.device)
            label = label.to(self.device)
            onehot_label = F.one_hot(label.squeeze().long(),num_classes=self.num_classes).float()


            '''Train Discriminator'''
            for d_step in range(self.n_critic):
                '''Noise and Fake Sequence'''
                noise = torch.FloatTensor(np.random.normal(0,1,(sequence.shape[0],self.latent_dim))).to(self.device)
                fake_label = torch.randint(0,self.num_classes,(sequence.shape[0],)).to(self.device)
                fake_sequence = self.generator(noise,fake_label)
                
                self.discriminator.zero_grad()
                real_out_adv, real_out_cls = self.discriminator(sequence)
                fake_out_adv, fake_out_cls = self.discriminator(fake_sequence)

                '''Compute Critic Loss (Temperary)'''
                # Gradient Penalty Loss
                alpha = torch.rand(sequence.size(0),1,1).to(self.device)
                x_hat = (alpha * sequence.data + (1-alpha) * fake_sequence).requires_grad_(True)
                out_src, _ = self.discriminator(x_hat)
                d_loss_gp = gradient_panelty(out_src,x_hat,device=self.device)

                # Wasserstein Loss
                d_real_loss = -torch.mean(real_out_adv)
                d_fake_loss = torch.mean(fake_out_adv)
                d_adv_loss = d_real_loss + d_fake_loss
                d_cls_loss = self.criterion(real_out_cls,onehot_label)
                d_loss = d_adv_loss + self.lambda_gp * d_loss_gp + self.lamdba_cls * d_cls_loss
                d_loss.backward()

                # Clip Weight
                nn.utils.clip_grad_norm_(self.discriminator.parameters(),10.)
                self.d_optimizer.step()

                self.writer.add_scalar('d_loss',d_loss.item(),iter*self.n_critic + d_step)
                tqdm.write(f"[Critic Step:{d_step}/{self.n_critic}][d_loss:{d_loss.item()}]")
                
            '''Train Generator'''
            self.generator.zero_grad()
            noise = torch.FloatTensor(np.random.normal(0,1,(sequence.shape[0],self.latent_dim))).to(self.device)
            fake_label = torch.randint(0,self.num_classes,(sequence.shape[0],)).to(self.device)
            fake_onehot_label = F.one_hot(fake_label.squeeze().long(),num_classes=self.num_classes).float()
            fake_sequence = self.generator(noise,fake_label)
            g_out_adv, g_out_cls = self.discriminator(fake_sequence)
            g_adv_loss = -torch.mean(g_out_adv)
            g_cls_loss = self.criterion(g_out_cls,fake_onehot_label)
            g_loss = g_adv_loss + self.lamdba_cls*g_cls_loss
            g_loss.backward()

            # Clip Weight
            nn.utils.clip_grad_norm_(self.generator.parameters(),10.)
            self.g_optimizer.step()

            self.writer.add_scalar('g_loss',g_loss.item(),iter)
            tqdm.write(f"[Iteration:{iter}/{self.max_iter}][g_loss:{g_loss.item()}]")
            
            
            '''LR scheduler step'''
            g_lr = self.g_scheduler.step(iter)
            d_lr = self.d_scheduler.step(iter)
            self.writer.add_scalar('LR/g_lr',g_lr,iter)
            self.writer.add_scalar('LR/d_lr',d_lr,iter)    


            if (iter+1) % self.save_iter == 0:
                '''Visualize SYnthetic data'''
                plot_buf = self.visualize(iter)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).unsqueeze(0)
                self.writer.add_image('Image',image[0],iter)

                self.save_weight(iter)



    def save_weight(self,epoch):
        ckpt = {
            'epoch':epoch+1,
            'gen_state_dict':self.generator.state_dict(),
            'dis_state_dict':self.discriminator.state_dict(),
            'gen_optim':self.g_optimizer.state_dict(),
            'dis_optim':self.d_optimizer.state_dict()
        }

        torch.save(ckpt,os.path.join(self.save_path,'checkpoint.pth'))


    
    def visualize(self,epoch):
        self.generator.eval()
        num_sample = 6
        noise = torch.FloatTensor(np.random.normal(0,1,(num_sample,self.latent_dim))).to(self.device)
        fake_label = torch.randint(0,self.num_classes,(num_sample,))
        fake_sequence = self.generator(noise,fake_label.to(self.device)).to('cpu').detach().numpy()
        _,c,_ = fake_sequence.shape

        fig, axs = plt.subplots(2,3,figsize=(20,8))
        fig.suptitle(f'Synthetic data at epoch {epoch}',fontsize=20)

        for i in range(2):
            for j in range(3):
                for k in range(c):
                    axs[i, j].plot(fake_sequence[i*3+j][k][:])
            
                axs[i, j].title.set_text(fake_label[i*3+j].item())

        buf = io.BytesIO()
        plt.savefig(buf,format='jpg')
        plt.close(fig)
        buf.seek(0)
        return buf


    def load_weight(self,checkpoint):
        ckpt = torch.load(checkpoint,map_location=self.device)
        self.generator.load_state_dict(ckpt['gen_state_dict'])
        self.discriminator.load_state_dict(ckpt['dis_state_dict'])
        self.g_optimizer.load_state_dict(ckpt['gen_optim'])
        self.d_optimizer.load_state_dict(ckpt['dis_optim'])

    


'''
Time Series Diffision Model: https://github.com/Y-debug-sys/Diffusion-TS
'''



# gaussian diffusion trainer class

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(
            self,
            backbone,
            seq_length,
            feature_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            eta=0.,
            use_ff=True,
            reg_weight=None,
            label_dim=None,
            configs=None,
            **kwargs
    ):
        super(Diffusion, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        self.label_dim = label_dim


        self.model = backbone

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate reweighting
        
        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, padding_masks=None,label=None):
        trend, season = self.model(x, t, padding_masks=padding_masks,label=label)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None,label=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, padding_masks,label=label)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True,label=None):
        _, x_start = self.model_predictions(x, t,label=label)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True,label=None):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised,label=label)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, shape,use_label=False):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        if use_label:
            label = F.one_hot(torch.randint(low=0,high=30,size=(shape[0],),device=device),num_classes=30).float()
        for t in reversed(range(0, self.num_timesteps)):
            img, _ = self.p_sample(img, t,label=label)

        return img, label if use_label else img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True,use_label=False):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)
        label = torch.randint(low=0,high=self.label_dim,size=(shape[0],),device=device) if use_label else None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised,label=label)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img, label if use_label else img

    def generate_mts(self, batch_size=16,use_label=False):
        feature_size, seq_length = self.feature_size, self.seq_length
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_length, feature_size),use_label=use_label)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None,label=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x.float(), t, padding_masks,label=label)

        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    
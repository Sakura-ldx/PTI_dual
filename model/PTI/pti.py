import torch
import torch.nn as nn
from lpips import LPIPS
from tqdm import tqdm

from model.PTI.criteria import l2_loss
from model.PTI.criteria.localitly_regulizer import Space_Regulizer
from model.PTI.projectors import z_plus_projector
from model.PTI.configs import hyperparameters


class PTI(nn.Module):
    def __init__(self, G, device):
        super(PTI, self).__init__()
        self.G = G
        self.device = device
        self.optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        self.pt_l2_lambda = hyperparameters.pt_l2_lambda
        self.pt_lpips_lambda = hyperparameters.pt_lpips_lambda
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(self.device).eval()
        self.space_regulizer = Space_Regulizer(self.G, self.lpips_loss)

        self.use_wandb = False

    def forward(self, z_plus_inits, images, is_finetune):
        z_pivots = []
        for image, z_plus_init in zip(images, z_plus_inits):
            id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255
            z_plus = z_plus_projector.project(self.G, id_image, device=torch.device(self.device), w_avg_samples=600,
                                              num_steps=450, initial_z=z_plus_init)
            z_pivots.append(z_plus)

        if is_finetune:
            for i in tqdm(range(550)):

                for image, z_pivot in zip(images, z_pivots):
                    real_images_batch = image.to(self.device)

                    generated_images, _ = self.G([z_pivot],
                                                 input_is_latent=False,
                                                 return_latents=False,
                                                 z_plus_latent=True)
                    loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch,
                                                                   self.G, False, z_pivot)

                    self.optimizer.zero_grad()
                    loss.requires_grad = True
                    loss.backward()
                    self.optimizer.step()

                    # use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                    # global_config.training_step += 1

        result_images = []
        for z_pivot in z_pivots:
            generated_images, _ = self.G([z_pivot],
                                         input_is_latent=False,
                                         return_latents=False,
                                         z_plus_latent=True)
            result_images.append(generated_images)

        return z_pivots, result_images

    def calc_loss(self, generated_images, real_images, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if self.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            loss += l2_loss_val * self.pt_l2_lambda
        if self.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * self.pt_lpips_lambda
        if use_ball_holder and self.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
import torch
import torch.nn.functional as F 

policy = 'color,translation,cutout' 

import torch.nn.functional as nnf
import random

#torch.autograd.set_detect_anomaly(True)
scaler = torch.cuda.amp.GradScaler() 

def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.mosaic_test = config.MOSAIC_TEST

		# loading previous weights
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(self.config.PATH, self.name + "_" + str(self.iteration) + "_gen.pth"))

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, os.path.join(self.config.PATH, self.name + "_" + str(self.iteration) + "_dis.pth"))


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1


        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        #real_scores = Discriminator(DiffAugment(reals, policy=policy))
        dis_real, dis_real_feat = self.discriminator(DiffAugment(dis_input_real, policy=policy))        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(DiffAugment(dis_input_fake, policy=policy))        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(DiffAugment(gen_input_fake, policy=policy))         # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss


        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss


        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.use_amp = config.USE_AMP

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        if(self.mosaic_test == 1):
          # resize image with random size. (256 currently hardcoded)
          mosaic_size = int(random.triangular(int(min(256*0.01, 256*0.01)), int(min(256*0.2, 256*0.2)), int(min(256*0.0625, 256*0.0625))))
          images_mosaic = nnf.interpolate(images, size=(mosaic_size, mosaic_size), mode='nearest')
          images_mosaic = nnf.interpolate(images_mosaic, size=(256, 256), mode='nearest')
          images_mosaic = (images * (1 - masks).float()) + (images_mosaic * (masks).float())
          outputs = self(images_mosaic, edges, masks)
        else:
          outputs = self(images, edges, masks)

        if self.use_amp == 1:
          with torch.cuda.amp.autocast(): 
            # process outputs
            #outputs = self(images, edges, masks)
            gen_loss = 0
            dis_loss = 0


            # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs.detach()
            #real_scores = Discriminator(DiffAugment(reals, policy=policy))
            dis_real, _ = self.discriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
            dis_fake, _ = self.discriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2


            # generator adversarial loss
            gen_input_fake = outputs
            #real_scores = Discriminator(DiffAugment(reals, policy=policy))
            gen_fake, _ = self.discriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss


            # generator l1 loss
            gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
            gen_loss += gen_l1_loss


            # generator perceptual loss
            gen_content_loss = self.perceptual_loss(outputs, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss


            # generator style loss
            gen_style_loss = self.style_loss(outputs * masks, images * masks)
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss
        else:
          # process outputs
          #outputs = self(images, edges, masks)
          gen_loss = 0
          dis_loss = 0


          # discriminator loss
          dis_input_real = images
          dis_input_fake = outputs.detach()
          #real_scores = Discriminator(DiffAugment(reals, policy=policy))
          dis_real, _ = self.discriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
          dis_fake, _ = self.discriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]
          dis_real_loss = self.adversarial_loss(dis_real, True, True)
          dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
          dis_loss += (dis_real_loss + dis_fake_loss) / 2


          # generator adversarial loss
          gen_input_fake = outputs
          #real_scores = Discriminator(DiffAugment(reals, policy=policy))
          gen_fake, _ = self.discriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
          gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
          gen_loss += gen_gan_loss


          # generator l1 loss
          gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
          gen_loss += gen_l1_loss


          # generator perceptual loss
          gen_content_loss = self.perceptual_loss(outputs, images)
          gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
          gen_loss += gen_content_loss


          # generator style loss
          gen_style_loss = self.style_loss(outputs * masks, images * masks)
          gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
          gen_loss += gen_style_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        if (self.mosaic_test == 1):
          # mosaic test
          images_masked = images
        else:
          images_masked = (images * (1 - masks).float()) + masks

        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        #dis_loss.backward(retain_graph = True)
        #gen_loss.backward()
        scaler.scale(dis_loss).backward(retain_graph = True) 
        scaler.scale(gen_loss).backward() 

        #self.gen_optimizer.step()
        #self.dis_optimizer.step()
        scaler.step(self.gen_optimizer) 
        scaler.step(self.dis_optimizer) 

        scaler.update() 

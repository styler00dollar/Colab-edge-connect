from .vic.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss
from .vic.filters import *
from .vic.colors import *
from .vic.discriminators import *
from .diffaug import *

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

            if self.config.DISCRIMINATOR == 'default':
              self.discriminator.load_state_dict(data['discriminator'])
            if self.config.DISCRIMINATOR == 'pixel':
              self.PixelDiscriminator.load_state_dict(data['discriminator'])
            if self.config.DISCRIMINATOR == 'patch':
              self.NLayerDiscriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(self.config.PATH, self.name + "_" + str(self.iteration) + "_gen.pth"))

        if self.config.DISCRIMINATOR == 'default':
          torch.save({
              #'discriminator': self.discriminator.state_dict()
              'discriminator': self.PixelDiscriminator.state_dict()
          }, os.path.join(self.config.PATH, self.name + "_" + str(self.iteration) + "_dis.pth"))
        if self.config.DISCRIMINATOR == 'pixel':
          torch.save({
              'discriminator': self.PixelDiscriminator.state_dict()
          }, os.path.join(self.config.PATH, self.name + "_" + str(self.iteration) + "_dis.pth"))
        if self.config.DISCRIMINATOR == 'patch':
          torch.save({
              'discriminator': self.NLayerDiscriminator.state_dict()
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
        """
        if(mosaic_size != None):
          # resize image with random size
          images_mosaic = nnf.interpolate(images, size=(mosaic_size, mosaic_size), mode='nearest')
          images_mosaic = nnf.interpolate(images_mosaic, size=(self.config.INPUT_SIZE, self.config.INPUT_SIZE), mode='nearest')
          images_mosaic = (images * (1 - masks).float()) + (images_mosaic * (masks).float())
          outputs = self(images_mosaic, edges, masks)
        else:
          outputs = self(images, edges, masks)

        """

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

        # original loss
        if self.use_amp == 1:
          with torch.cuda.amp.autocast():
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

        else:
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

    def forward(self, images, edges, masks, mosaic_size=None):
        if(mosaic_size != None):
          # resize image with random size
          images_mosaic = nnf.interpolate(images, size=(mosaic_size, mosaic_size), mode='nearest')
          images_mosaic = nnf.interpolate(images_mosaic, size=(self.config.INPUT_SIZE, self.config.INPUT_SIZE), mode='nearest')
          images_mosaic = (images * (1 - masks).float()) + (images_mosaic * (masks).float())
          images_masked = self(images_mosaic, edges, masks)
        else:
          images_masked = (edges * (1 - masks))

        edges_masked = (edges * (1 - masks))
        #images_masked = (images * (1 - masks)) + masks
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
        """
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)
        """
        # original loss
        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.generator_loss = config.GENERATOR_LOSS

        self.add_module('generator', generator)

        if self.config.DISCRIMINATOR == 'default':
          discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
          self.add_module('discriminator', discriminator)

        if self.config.DISCRIMINATOR == 'pixel':
          _PixelDiscriminator = PixelDiscriminator(input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d)
          self.add_module('PixelDiscriminator', _PixelDiscriminator)

        if self.config.DISCRIMINATOR == 'patch':
          _NLayerDiscriminator = NLayerDiscriminator(input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d)
          self.add_module('NLayerDiscriminator', _NLayerDiscriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # new added loss
        # CharbonnierLoss (L1) (already implemented?)
        _CharbonnierLoss = CharbonnierLoss()
        self.add_module('_CharbonnierLoss', _CharbonnierLoss)
        # GANLoss (vanilla, lsgan, srpgan, nsgan, hinge, wgan-gp)
        _GANLoss = GANLoss('vanilla', real_label_val=1.0, fake_label_val=0.0)
        self.add_module('_GANLoss', _GANLoss)
        # GradientPenaltyLoss
        _GradientPenaltyLoss = GradientPenaltyLoss()
        self.add_module('_GradientPenaltyLoss', _GradientPenaltyLoss)
        # HFENLoss
        #l_hfen_type = CharbonnierLoss() # nn.L1Loss(), nn.MSELoss(), CharbonnierLoss(), ElasticLoss(), RelativeL1(), L1CosineSim()
        if self.config.HFEN_TYPE == 'L1':
          l_hfen_type = nn.L1Loss()
        if self.config.HFEN_TYPE == 'MSE':
          l_hfen_type = nn.MSELoss()
        if self.config.HFEN_TYPE == 'Charbonnier':
          l_hfen_type = CharbonnierLoss()
        if self.config.HFEN_TYPE == 'ElasticLoss':
          l_hfen_type = ElasticLoss()
        if self.config.HFEN_TYPE == 'RelativeL1':
          l_hfen_type = RelativeL1()
        if self.config.HFEN_TYPE == 'L1CosineSim':
          l_hfen_type = L1CosineSim()

        _HFENLoss = HFENLoss(loss_f=l_hfen_type, kernel='log', kernel_size=15, sigma = 2.5, norm = False)
        self.add_module('_HFENLoss', _HFENLoss)
        # TVLoss
        _TVLoss = TVLoss(tv_type='tv', p = 1)
        self.add_module('_TVLoss', _TVLoss)
        # GradientLoss
        _GradientLoss = GradientLoss(loss_f = None, reduction='mean', gradientdir='2d')
        self.add_module('_GradientLoss', _GradientLoss)
        # ElasticLoss
        _ElasticLoss = ElasticLoss(a=0.2, reduction='mean')
        self.add_module('_ElasticLoss', _ElasticLoss)
        # RelativeL1 (todo?)
        _RelativeL1 = RelativeL1(eps=.01, reduction='mean')
        self.add_module('_RelativeL1', _RelativeL1)
        # L1CosineSim
        _L1CosineSim = L1CosineSim(loss_lambda=5, reduction='mean')
        self.add_module('_L1CosineSim', _L1CosineSim)
        # ClipL1
        _ClipL1 = ClipL1(clip_min=0.0, clip_max=10.0)
        self.add_module('_ClipL1', _ClipL1)
        # FFTloss
        _FFTloss = FFTloss(loss_f = torch.nn.L1Loss, reduction='mean')
        self.add_module('_FFTloss', _FFTloss)
        # OFLoss
        _OFLoss = OFLoss()
        self.add_module('_OFLoss', _OFLoss)
        # ColorLoss (untested)
        ds_f = torch.nn.AvgPool2d=((3, 3)) # kernel_size=5
        _ColorLoss = ColorLoss(loss_f = torch.nn.L1Loss, reduction='mean', ds_f=ds_f)
        self.add_module('_ColorLoss', _ColorLoss)
        # GPLoss
        _GPLoss = GPLoss(trace=False, spl_denorm=False)
        self.add_module('_GPLoss', _GPLoss)
        # CPLoss (SPL_ComputeWithTrace, SPLoss)
        _CPLoss = CPLoss(rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False)
        self.add_module('_CPLoss', _CPLoss)
        # Contextual_Loss
        layers_weights = {'conv_1_1': 1.0, 'conv_3_2': 1.0}
        _Contextual_Loss = Contextual_Loss(layers_weights, crop_quarter=False, max_1d_size=100,
            distance_type = 'cosine', b=1.0, band_width=0.5,
            use_vgg = True, net = 'vgg19', calc_type = 'regular')
        self.add_module('_Contextual_Loss', _Contextual_Loss)

        """
        if self.config.DISCRIMINATOR == 'pixel':
          pixel_criterion = torch.nn.BCEWithLogitsLoss()
          self.add_module('pixel_criterion', pixel_criterion)

        if self.config.DISCRIMINATOR == 'patch':
          patch_criterion = torch.nn.BCEWithLogitsLoss()
          self.add_module('patch_criterion', patch_criterion)
        """

        if self.config.DISCRIMINATOR_CALC == 'BCEWithLogitsLoss':
          bce_criterion = nn.BCEWithLogitsLoss()
          self.add_module('bce_criterion', bce_criterion)

        if self.config.DISCRIMINATOR_CALC == 'MSELoss':
          mse_criterion = nn.MSELoss()
          self.add_module('mse_criterion', mse_criterion)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        if self.config.DISCRIMINATOR == 'default':
          self.dis_optimizer = optim.Adam(
              params=discriminator.parameters(),
              #params=self.PixelDiscriminator.parameters(),
              lr=float(config.LR) * float(config.D2G_LR),
              betas=(config.BETA1, config.BETA2)
          )
        if self.config.DISCRIMINATOR == 'pixel':
          self.dis_optimizer = optim.Adam(
              #params=discriminator.parameters(),
              params=self.PixelDiscriminator.parameters(),
              lr=float(config.LR) * float(config.D2G_LR),
              betas=(config.BETA1, config.BETA2)
          )


        if self.config.DISCRIMINATOR == 'patch':
          self.dis_optimizer = optim.Adam(
              #params=discriminator.parameters(),
              params=self.NLayerDiscriminator.parameters(),
              lr=float(config.LR) * float(config.D2G_LR),
              betas=(config.BETA1, config.BETA2)
          )




        self.use_amp = config.USE_AMP

    def process(self, images, edges, masks, mosaic_size=None):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        if(mosaic_size != None):
          # resize image with random size
          images_mosaic = nnf.interpolate(images, size=(mosaic_size, mosaic_size), mode='nearest')
          images_mosaic = nnf.interpolate(images_mosaic, size=(self.config.INPUT_SIZE, self.config.INPUT_SIZE), mode='nearest')
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

            if self.config.DISCRIMINATOR == 'default':
              dis_real, _ = self.discriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
              dis_fake, _ = self.discriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]

            if self.config.DISCRIMINATOR == 'pixel':
              dis_real = self.PixelDiscriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
              dis_fake = self.PixelDiscriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]

            if self.config.DISCRIMINATOR == 'patch':
              dis_real = self.NLayerDiscriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
              dis_fake = self.NLayerDiscriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]



            if self.config.DISCRIMINATOR_CALC == 'BCEWithLogitsLoss':
              dis_fake_loss = self.bce_criterion(dis_fake, torch.ones_like(dis_fake))
              dis_real_loss = self.bce_criterion(dis_real, torch.zeros_like(dis_real))

            if self.config.DISCRIMINATOR_CALC == 'MSELoss':
              dis_fake_loss = self.mse_criterion(dis_fake, torch.ones_like(dis_fake))
              dis_real_loss = self.mse_criterion(dis_real, torch.zeros_like(dis_real))


            dis_loss += ((dis_real_loss * self.config.DISCRIMINATOR_REAL_LOSS_WEIGHT) + (dis_fake_loss * self.config.DISCRIMINATOR_FAKE_LOSS_WEIGHT)) / 2


            # original generator loss
            # generator adversarial loss
            gen_input_fake = outputs

            if 'DEFAULT_GAN' in self.generator_loss:
              if self.config.DISCRIMINATOR == 'default':
                gen_fake, _ = self.discriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
              if self.config.DISCRIMINATOR == 'pixel':
                gen_fake = self.PixelDiscriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
              if self.config.DISCRIMINATOR == 'patch':
                gen_fake = self.NLayerDiscriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]

              #gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT

              if self.config.GENERATOR_CALC == 'BCEWithLogitsLoss':
                gen_gan_loss = self.bce_criterion(gen_fake, torch.ones_like(gen_fake))

              if self.config.GENERATOR_CALC == 'MSELoss':
                gen_gan_loss = self.mse_criterion(gen_fake, torch.ones_like(gen_fake))

              gen_loss += gen_gan_loss * self.config.GENERATOR_CALC_WEIGHT

            # generator l1 loss
            if 'DEFAULT_L1' in self.generator_loss:
              gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
              gen_loss += gen_l1_loss

            # generator perceptual loss
            if 'Perceptual' in self.generator_loss:
              gen_content_loss = self.perceptual_loss(outputs, images)
              gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
              gen_loss += gen_content_loss

            # generator style loss
            if 'Style' in self.generator_loss:
              gen_style_loss = self.style_loss(outputs * masks, images * masks)
              gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
              gen_loss += gen_style_loss


            # new loss
            # CharbonnierLoss (L1) (already implemented?)
            if 'NEW_L1' in self.generator_loss:
              gen_loss += self.config.L1_LOSS_WEIGHT * self._CharbonnierLoss(outputs, images)

            # GANLoss (vanilla, lsgan, srpgan, nsgan, hinge, wgan-gp)
            if 'NEW_GAN' in self.generator_loss:
              gen_loss += self.config.NEW_GAN_WEIGHT * self._GANLoss(outputs, images)
            # GradientPenaltyLoss
            #gen_loss += self._GradientPenaltyLoss(outputs, images, interp_crit) # not sure what interp_crit is
            # HFENLoss
            if 'HFEN' in self.generator_loss:
              gen_loss += self.config.HFEN_WEIGHT * self._HFENLoss(outputs, images)
            # TVLoss
            if 'TV' in self.generator_loss:
              gen_loss += self.config.TV_WEIGHT * self._TVLoss(outputs)
            # GradientLoss
            #gen_loss += self._GradientLoss(outputs, images) # TypeError: 'NoneType' object is not callable
            # ElasticLoss
            if 'ElasticLoss' in self.generator_loss:
              gen_loss += self.config.ElasticLoss_WEIGHT * self._ElasticLoss(outputs, images)
            # RelativeL1 (todo?)
            if 'RelativeL1' in self.generator_loss:
              gen_loss += self.config.RelativeL1_WEIGHT * self._RelativeL1(outputs, images)
            # L1CosineSim
            if 'L1CosineSim' in self.generator_loss:
              gen_loss += self.config.L1CosineSim_WEIGHT * self._L1CosineSim(outputs, images)
            # ClipL1
            if 'ClipL1' in self.generator_loss:
              gen_loss += self.config.ClipL1_WEIGHT * self._ClipL1(outputs, images)
            # FFTloss
            if 'FFT' in self.generator_loss:
              gen_loss += self.config.FFT_WEIGHT * self._FFTloss(outputs, images)
            # OFLoss
            if 'OF' in self.generator_loss:
              gen_loss += self.config.OF_WEIGHT * self._OFLoss(outputs)
            # ColorLoss (untested)
            #gen_loss += self._ColorLoss(outputs, images) # TypeError: 'NoneType' object is not callable
            # GPLoss
            if 'GP' in self.generator_loss:
              gen_loss += self.config.GP_WEIGHT * self._GPLoss(outputs, images)
            # CPLoss (SPL_ComputeWithTrace, SPLoss)
            if 'CP' in self.generator_loss:
              gen_loss += self.config.CP_WEIGHT * self._CPLoss(outputs, images)
            # Contextual_Loss
            if 'Contextual' in self.generator_loss:
              gen_loss += self.config.Contextual_WEIGHT * self._Contextual_Loss(outputs, images)

        else:
          # process outputs
          #outputs = self(images, edges, masks)
          gen_loss = 0
          dis_loss = 0


          # discriminator loss
          dis_input_real = images
          dis_input_fake = outputs.detach()
          #real_scores = Discriminator(DiffAugment(reals, policy=policy))

          if self.config.DISCRIMINATOR == 'default':
            dis_real, _ = self.discriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
            dis_fake, _ = self.discriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]

          if self.config.DISCRIMINATOR == 'pixel':
            dis_real = self.PixelDiscriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
            dis_fake = self.PixelDiscriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]

          if self.config.DISCRIMINATOR == 'patch':
            dis_real = self.NLayerDiscriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
            dis_fake = self.NLayerDiscriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]



          if self.config.DISCRIMINATOR_CALC == 'BCEWithLogitsLoss':
            dis_fake_loss = self.bce_criterion(dis_fake, torch.ones_like(dis_fake))
            dis_real_loss = self.bce_criterion(dis_real, torch.zeros_like(dis_real))

          if self.config.DISCRIMINATOR_CALC == 'MSELoss':
            dis_fake_loss = self.mse_criterion(dis_fake, torch.ones_like(dis_fake))
            dis_real_loss = self.mse_criterion(dis_real, torch.zeros_like(dis_real))


          dis_loss += ((dis_real_loss * self.config.DISCRIMINATOR_REAL_LOSS_WEIGHT) + (dis_fake_loss * self.config.DISCRIMINATOR_FAKE_LOSS_WEIGHT)) / 2


          # original generator loss
          # generator adversarial loss
          gen_input_fake = outputs

          if 'DEFAULT_GAN' in self.generator_loss:
            if self.config.DISCRIMINATOR == 'default':
              gen_fake, _ = self.discriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
            if self.config.DISCRIMINATOR == 'pixel':
              gen_fake = self.PixelDiscriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
            if self.config.DISCRIMINATOR == 'patch':
              gen_fake = self.NLayerDiscriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]

            #gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT

            if self.config.GENERATOR_CALC == 'BCEWithLogitsLoss':
              gen_gan_loss = self.bce_criterion(gen_fake, torch.ones_like(gen_fake))

            if self.config.GENERATOR_CALC == 'MSELoss':
              gen_gan_loss = self.mse_criterion(gen_fake, torch.ones_like(gen_fake))

            gen_loss += gen_gan_loss * self.config.GENERATOR_CALC_WEIGHT

          # generator l1 loss
          if 'DEFAULT_L1' in self.generator_loss:
            gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
            gen_loss += gen_l1_loss

          # generator perceptual loss
          if 'Perceptual' in self.generator_loss:
            gen_content_loss = self.perceptual_loss(outputs, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss

          # generator style loss
          if 'Style' in self.generator_loss:
            gen_style_loss = self.style_loss(outputs * masks, images * masks)
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss


          # new loss
          # CharbonnierLoss (L1) (already implemented?)
          if 'NEW_L1' in self.generator_loss:
            gen_loss += self.config.L1_LOSS_WEIGHT * self._CharbonnierLoss(outputs, images)

          # GANLoss (vanilla, lsgan, srpgan, nsgan, hinge, wgan-gp)
          if 'NEW_GAN' in self.generator_loss:
            gen_loss += self.config.NEW_GAN_WEIGHT * self._GANLoss(outputs, images)
          # GradientPenaltyLoss
          #gen_loss += self._GradientPenaltyLoss(outputs, images, interp_crit) # not sure what interp_crit is
          # HFENLoss
          if 'HFEN' in self.generator_loss:
            gen_loss += self.config.HFEN_WEIGHT * self._HFENLoss(outputs, images)
          # TVLoss
          if 'TV' in self.generator_loss:
            gen_loss += self.config.TV_WEIGHT * self._TVLoss(outputs)
          # GradientLoss
          #gen_loss += self._GradientLoss(outputs, images) # TypeError: 'NoneType' object is not callable
          # ElasticLoss
          if 'ElasticLoss' in self.generator_loss:
            gen_loss += self.config.ElasticLoss_WEIGHT * self._ElasticLoss(outputs, images)
          # RelativeL1 (todo?)
          if 'RelativeL1' in self.generator_loss:
            gen_loss += self.config.RelativeL1_WEIGHT * self._RelativeL1(outputs, images)
          # L1CosineSim
          if 'L1CosineSim' in self.generator_loss:
            gen_loss += self.config.L1CosineSim_WEIGHT * self._L1CosineSim(outputs, images)
          # ClipL1
          if 'ClipL1' in self.generator_loss:
            gen_loss += self.config.ClipL1_WEIGHT * self._ClipL1(outputs, images)
          # FFTloss
          if 'FFT' in self.generator_loss:
            gen_loss += self.config.FFT_WEIGHT * self._FFTloss(outputs, images)
          # OFLoss
          if 'OF' in self.generator_loss:
            gen_loss += self.config.OF_WEIGHT * self._OFLoss(outputs)
          # ColorLoss (untested)
          #gen_loss += self._ColorLoss(outputs, images) # TypeError: 'NoneType' object is not callable
          # GPLoss
          if 'GP' in self.generator_loss:
            gen_loss += self.config.GP_WEIGHT * self._GPLoss(outputs, images)
          # CPLoss (SPL_ComputeWithTrace, SPLoss)
          if 'CP' in self.generator_loss:
            gen_loss += self.config.CP_WEIGHT * self._CPLoss(outputs, images)
          # Contextual_Loss
          if 'Contextual' in self.generator_loss:
            gen_loss += self.config.Contextual_WEIGHT * self._Contextual_Loss(outputs, images)


        """
        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]
        """
        logs = [] # txt logs currently unsupported
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

import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import rescale
from .utils import create_mask
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        self.random_crop = config.RANDOM_CROP
        self.horizontal_flip = config.HORIZONTAL_FLIP
        self.horizontal_flip_ratio = config.HORIZONTAL_FLIP_RATIO
        self.vertical_flip = config.VERTICAL_FLIP
        self.vertical_flip_ratio = config.VERTICAL_FLIP_RATIO

        self.config = config

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize

        # rescale image to minimum needed INPUT_SIZE (keeping aspect ratio) (not working)
        """
        if img.shape[0] < size or img.shape[1] < size:
            imgh, imgw = img.shape[0:2]
            small_side = np.minimum(imgh, imgw)
            img = rescale(img, (size/small_side)*1.1, anti_aliasing=False)
        """

        # if image is smaller than INPUT_SIZE, then resize to needed dimension
        if img.shape[0] < size or img.shape[1] < size:
            img = cv2.resize(img, (size, size), cv2.INTER_NEAREST)
        else:
          #if img.shape[0] != size or img.shape[1] != size:
          if size != 0 and self.config.RESIZE_MODE == 'random_resize':
              imgh, imgw = img.shape[0:2]
              small_side = np.minimum(imgh, imgw)
              random_resize_ratio = np.random.randint(self.config.INPUT_SIZE, small_side, size=1) / small_side
              img = cv2.resize(img, (img.shape[0]*random_resize_ratio, img.shape[1]*random_resize_ratio), cv2.INTER_NEAREST)

          if size != 0 and self.config.RESIZE_MODE == 'input_size_and_random_downscale':
              if random.uniform(0, 1) < self.config.INPUT_SIZE_AND_RANDOM_DOWNSCALE_RATIO:
                img = self.resize(img, size, size)
              else:
                random_resize_ratio = np.random.randint(self.config.INPUT_SIZE, img.shape[0], size=1) / img.shape[0]
                img = cv2.resize(img, (img.shape[0]*random_resize_ratio, img.shape[1]*random_resize_ratio), cv2.INTER_NEAREST)

          # crop
          # randomly crop data
          if size != 0 and self.config.CROP_MODE == 'random_crop':
              x = random.randint(0, img.shape[1] - size)
              y = random.randint(0, img.shape[0] - size)
              img = img[y:y+size, x:x+size]

          if size != 0 and self.config.CROP_MODE == 'center_crop':
            imgh, imgw = img.shape[0:2]
            if centerCrop and imgh != imgw:
                side = np.minimum(imgh, imgw)
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img[j:j + side, i:i + side, ...]
            img = scipy.misc.imresize(img, [size, size])



        # flip with numpy
        if self.horizontal_flip == 1:
          if random.randint(0, 1) < self.horizontal_flip_ratio:
            img = np.flip(img, 1)

        if self.vertical_flip == 1:
          if random.randint(0, 1) < self.vertical_flip_ratio:
            img = np.flip(img, 0)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    """
    def resize(self, img, height, width, centerCrop=True):
        if self.random_crop == 1:
          # randomly crop data
          x = random.randint(0, img.shape[1] - width)
          y = random.randint(0, img.shape[0] - height)
          img = img[y:y+height, x:x+width]
        else:
          # Original center crop with resize
          imgh, imgw = img.shape[0:2]
          if centerCrop and imgh != imgw:
              # center crop
              side = np.minimum(imgh, imgw)
              j = (imgh - side) // 2
              i = (imgw - side) // 2
              img = img[j:j + side, i:i + side, ...]
          img = scipy.misc.imresize(img, [height, width])
        return img
    """
    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, delimiter='\n', encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

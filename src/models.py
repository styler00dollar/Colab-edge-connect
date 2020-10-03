# victorca25's loss implementations
# https://github.com/victorca25/BasicSR/blob/dev2/codes/models/modules/loss.py
import torch
import torch.nn as nn
import math
import numbers
import torch.nn.functional as F
import numpy as np

import torchvision.models.vgg as vgg
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

################################################################################################################################################################################################
# VGG MODEL
# models.modules.architectures.perceptual

vgg_layer19 = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16, 'pool_3': 18, 'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25, 'pool_4': 27, 'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34, 'pool_5': 36
}
vgg_layer_inv19 = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'conv_3_4', 18: 'pool_3', 19: 'conv_4_1', 21: 'conv_4_2', 23: 'conv_4_3', 25: 'conv_4_4', 27: 'pool_4', 28: 'conv_5_1', 30: 'conv_5_2', 32: 'conv_5_3', 34: 'conv_5_4', 36: 'pool_5'
}
# VGG 16 layers to listen to
vgg_layer16 = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'pool_3': 16, 'conv_4_1': 17, 'conv_4_2': 19, 'conv_4_3': 21, 'pool_4': 23, 'conv_5_1': 24, 'conv_5_2': 26, 'conv_5_3': 28, 'pool_5': 30
}
vgg_layer_inv16 = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'pool_3', 17: 'conv_4_1', 19: 'conv_4_2', 21: 'conv_4_3', 23: 'pool_4', 24: 'conv_5_1', 26: 'conv_5_2', 28: 'conv_5_3', 30: 'pool_5'
}

class VGG_Model(nn.Module):
    """
        A VGG model with listerners in the layers. 
        Will return a dictionary of outputs that correspond to the 
        layers set in "listen_list".
    """
    def __init__(self, listen_list=None, net='vgg19', use_input_norm=True, z_norm=False):
        super(VGG_Model, self).__init__()
        #vgg = vgg16(pretrained=True)
        if net == 'vgg19':
            vgg_net = vgg.vgg19(pretrained=True)
            vgg_layer = vgg_layer19
            self.vgg_layer_inv = vgg_layer_inv19
        elif net == 'vgg16':
            vgg_net = vgg.vgg16(pretrained=True)
            vgg_layer = vgg_layer16
            self.vgg_layer_inv = vgg_layer_inv16
        self.vgg_model = vgg_net.features
        self.use_input_norm = use_input_norm
        # image normalization
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.tensor(
                    [[[0.485-1]], [[0.456-1]], [[0.406-1]]], requires_grad=False)
                std = torch.tensor(
                    [[[0.229*2]], [[0.224*2]], [[0.225*2]]], requires_grad=False)
            else: # input in range [0,1]
                mean = torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
                std = torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        vgg_dict = vgg_net.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean.detach()) / self.std.detach()

        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[self.vgg_layer_inv[index]] = x
        return self.features

################################################################################################################################################################################################
# from dataops.filters import *
# codes/dataops/filters.py

'''
    Multiple image filters used by different functions. Can also be used as augmentations.
'''

import numbers
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from dataops.common import denorm

def denorm(x, min_max=(-1.0, 1.0)):
    '''
        Denormalize from [-1,1] range to [0,1]
        formula: xi' = (xi - mu)/sigma
        Example: "out = (x + 1.0) / 2.0" for denorm 
            range (-1,1) to (0,1)
        for use with proper act in Generator output (ie. tanh)
    '''
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    if isinstance(x, torch.Tensor):
        return out.clamp(0, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")


def get_kernel_size(sigma = 6):
    '''
        Get optimal gaussian kernel size according to sigma * 6 criterion 
        (must return an int)
        Alternative from Matlab: kernel_size=2*np.ceil(3*sigma)+1
        https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
    '''
    kernel_size = np.ceil(sigma*6)
    return kernel_size

def get_kernel_sigma(kernel_size = 5):
    '''
        Get optimal gaussian kernel sigma (variance) according to kernel_size/6 
        Alternative from Matlab: sigma = (kernel_size-1)/6
    '''
    return kernel_size/6.0

def get_kernel_mean(kernel_size = 5):
    '''
        Get gaussian kernel mean
    '''
    return (kernel_size - 1) / 2.0

def kernel_conv_w(kernel, channels: int =3):
    '''
        Reshape a H*W kernel to 2d depthwise convolutional 
            weight (for loading in a Conv2D)
    '''

    # Dynamic window expansion. expand() does not copy memory, needs contiguous()
    kernel = kernel.expand(channels, 1, *kernel.size()).contiguous()
    return kernel

#@torch.jit.script
def get_gaussian_kernel1d(kernel_size: int,
                sigma: float = 1.5, 
                #channel: int = None,
                force_even: bool = False) -> torch.Tensor:
    r"""Function that returns 1-D Gaussian filter kernel coefficients.

    Args:
        kernel_size (int): filter/window size. It should be odd and positive.
        sigma (float): gaussian standard deviation, sigma of normal distribution
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        torch.Tensor: 1D tensor with 1D gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
        
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )

    if kernel_size % 2 == 0:
        x = torch.arange(kernel_size).float() - kernel_size // 2    
        x = x + 0.5
        gauss = torch.exp((-x.pow(2.0) / float(2 * sigma ** 2)))
    else: #much faster
        gauss = torch.Tensor([np.exp(-(x - kernel_size//2)**2/float(2*sigma**2)) for x in range(kernel_size)])

    gauss /= gauss.sum()
    
    return gauss

#To get the kernel coefficients
def get_gaussian_kernel2d(
        #kernel_size: Tuple[int, int],
        kernel_size,
        #sigma: Tuple[float, float],
        sigma,
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
         Modified with a faster kernel creation if the kernel size
         is odd. 
    Args:
        kernel_size (Tuple[int, int]): filter (window) sizes in the x and y 
         direction. Sizes should be odd and positive, unless force_even is
         used.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """

    if isinstance(kernel_size, (int, float)): 
        kernel_size = (kernel_size, kernel_size)

    if isinstance(sigma, (int, float)): 
        sigma = (sigma, sigma)

    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    
    return kernel_2d

def get_gaussian_kernel(kernel_size=5, sigma=3, dim=2):
    '''
        This function can generate gaussian kernels in any dimension,
            but its 3 times slower than get_gaussian_kernel2d()
    Arguments:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
            Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
            direction.
        dim: the image dimension (2D=2, 3D=3, etc)
    Returns:
        Tensor: tensor with gaussian filter matrix coefficients.
    '''

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    kernel = kernel / torch.sum(kernel)    
    return kernel

#TODO: could be modified to generate kernels in different dimensions
def get_box_kernel(kernel_size: int = 5, dim=2):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim

    kx: float=  float(kernel_size[0])
    ky: float=  float(kernel_size[1])
    box_kernel = torch.Tensor(np.ones((kx, ky)) / (kx*ky))

    return box_kernel



#TODO: Can change HFEN to use either LoG, DoG or XDoG
def get_log_kernel_5x5():
    '''
    This is a precomputed LoG kernel that has already been convolved with
    Gaussian, for edge detection. 
    
    http://fourier.eng.hmc.edu/e161/lectures/gradient/node8.html
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
    The 2-D LoG can be approximated by a 5 by 5 convolution kernel such as:
    weight_log = torch.Tensor([
                    [0, 0, 1, 0, 0],
                    [0, 1, 2, 1, 0],
                    [1, 2, -16, 2, 1],
                    [0, 1, 2, 1, 0],
                    [0, 0, 1, 0, 0]
                ])
    This is an approximate to the LoG kernel with kernel size 5 and optimal 
    sigma ~6 (0.590155...).
    '''
    return torch.Tensor([
                [0, 0, 1, 0, 0],
                [0, 1, 2, 1, 0],
                [1, 2, -16, 2, 1],
                [0, 1, 2, 1, 0],
                [0, 0, 1, 0, 0]
            ])

#dim is the image dimension (2D=2, 3D=3, etc), but for now the final_kernel is hardcoded to 2D images
#Not sure if it would make sense in higher dimensions
#Note: Kornia suggests their laplacian kernel can also be used to generate LoG kernel: 
# https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
def get_log_kernel2d(kernel_size=5, sigma=None, dim=2): #sigma=0.6; kernel_size=5
    
    #either kernel_size or sigma are required:
    if not kernel_size and sigma:
        kernel_size = get_kernel_size(sigma)
        kernel_size = [kernel_size] * dim #note: should it be [kernel_size] or [kernel_size-1]? look below 
    elif kernel_size and not sigma:
        sigma = get_kernel_sigma(kernel_size)
        sigma = [sigma] * dim

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size-1] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    grids = torch.meshgrid([torch.arange(-size//2,size//2+1,1) for size in kernel_size])

    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, grids):
        kernel *= torch.exp(-(mgrid**2/(2.*std**2)))
    
    #TODO: For now hardcoded to 2 dimensions, test to make it work in any dimension
    final_kernel = (kernel) * ((grids[0]**2 + grids[1]**2) - (2*sigma[0]*sigma[1])) * (1/((2*math.pi)*(sigma[0]**2)*(sigma[1]**2)))
    
    #TODO: Test if normalization has to be negative (the inverted peak should not make a difference)
    final_kernel = -final_kernel / torch.sum(final_kernel)
    
    return final_kernel

def get_log_kernel(kernel_size: int = 5, sigma: float = None, dim: int = 2):
    '''
        Returns a Laplacian of Gaussian (LoG) kernel. If the kernel is known, use it,
        else, generate a kernel with the parameters provided (slower)
    '''
    if kernel_size ==5 and not sigma and dim == 2: 
        return get_log_kernel_5x5()
    else:
        return get_log_kernel2d(kernel_size, sigma, dim)

#TODO: use
# Implementation of binarize operation (for edge detectors)
def binarize(bin_img, threshold):
  #bin_img = img > threshold
  bin_img[bin_img < threshold] = 0.
  return bin_img




def get_laplacian_kernel_3x3(alt=False) -> torch.Tensor:
    """
        Utility function that returns a laplacian kernel of 3x3
            https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
            http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
        
        This is called a negative Laplacian because the central peak is negative. 
        It is just as appropriate to reverse the signs of the elements, using 
        -1s and a +4, to get a positive Laplacian. It doesn't matter:

        laplacian_kernel = torch.Tensor([
                                    [0,  -1, 0],
                                    [-1, 4, -1],
                                    [0,  -1, 0]
                                ])

        Alternative Laplacian kernel as produced by Kornia (this is positive Laplacian,
        like: https://kornia.readthedocs.io/en/latest/filters.html
        laplacian_kernel = torch.Tensor([
                                    [-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]
                                ])

    """
    if alt:
        return torch.tensor([
                    [-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]
                ])
    else:
        return torch.tensor([
                    [0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0],
                ])

def get_gradient_kernel_3x3() -> torch.Tensor:
    """
        Utility function that returns a gradient kernel of 3x3
            in x direction (transpose for y direction)
            kernel_gradient_v = [[0, -1, 0], 
                                 [0, 0, 0], 
                                 [0, 1, 0]]
            kernel_gradient_h = [[0, 0, 0], 
                                 [-1, 0, 1], 
                                 [0, 0, 0]]
    """
    return torch.tensor([
                   [0, 0, 0], 
                   [-1, 0, 1], 
                   [0, 0, 0],
            ])

def get_scharr_kernel_3x3() -> torch.Tensor:
    """
        Utility function that returns a scharr kernel of 3x3
            in x direction (transpose for y direction)
    """
    return torch.tensor([
                   [-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3],
    ])

def get_prewitt_kernel_3x3() -> torch.Tensor:
    """
        Utility function that returns a prewitt kernel of 3x3
            in x direction (transpose for y direction).
        
        Prewitt in x direction: This mask is called the 
            (vertical) Prewitt Edge Detector
            prewitt_x= np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])
        
        Prewitt in y direction: This mask is called the 
            (horizontal) Prewitt Edge Detector
            prewitt_y= np.array([[-1,-1,-1],
                                 [0, 0, 0],
                                 [1, 1, 1]])

        Note that a Prewitt operator is a 1D box filter convolved with 
            a derivative operator 
            finite_diff = [-1, 0, 1]
            simple_box = [1, 1, 1]

    """
    return torch.tensor([
                   [-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1],
    ])

#https://github.com/kornia/kornia/blob/master/kornia/filters/kernels.py
def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3
        sobel in x direction
            sobel_x= np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
        sobel in y direction
            sobel_y= np.array([[-1,-2,-1],
                               [0, 0, 0],
                               [1, 2, 1]])
        
        Note that a Sobel operator is a [1 2 1] filter convolved with 
            a derivative operator.
            finite_diff = [1, 2, 1]
            simple_box = [1, 1, 1]
    """
    return torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])

#https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
def get_sobel_kernel_2d(kernel_size=3):
    # get range
    range = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # compute a grid the numerator and the axis-distances
    y, x = torch.meshgrid(range, range)
    #Note: x is edge detector in x, y is edge detector in y, if not dividing by den
    den = (x ** 2 + y ** 2)
    #den[:, kernel_size // 2] = 1  # avoid division by zero at the center of den
    den[kernel_size // 2, kernel_size // 2] = 1  # avoid division by zero at the center of den
    #sobel_2D = x / den #produces kernel in range (0,1)
    sobel_2D = 2*x / den #produces same kernel as kornia
    return sobel_2D

def get_sobel_kernel(kernel_size=3):
    '''
    Sobel kernel
        https://en.wikipedia.org/wiki/Sobel_operator
    Note: using the Sobel filters needs two kernels, one in X axis and one in Y 
        axis (which is the transpose of X), to get the gradients in both directions.
        The same kernel can be used in both cases.
    '''
    if kernel_size==3:
        return get_sobel_kernel_3x3()
    else:
        return get_sobel_kernel_2d(kernel_size)



#To apply the 1D filter in X and Y axis (For SSIM)
#@torch.jit.script
def apply_1Dfilter(input, win, use_padding: bool=False):  
    r""" Apply 1-D kernel to input in X and Y axes.
         Separable filters like the Gaussian blur can be applied to 
         a two-dimensional image as two independent one-dimensional 
         calculations, so a 2-dimensional convolution operation can 
         be separated into two 1-dimensional filters. This reduces 
         the cost of computing the operator.
           https://en.wikipedia.org/wiki/Separable_filter
    Args:
        input (torch.Tensor): a batch of tensors to be filtered
        window (torch.Tensor): 1-D gauss kernel
        use_padding: padding image before conv
    Returns:
        torch.Tensor: filtered tensors
    """
    #N, C, H, W = input.shape
    C = input.shape[1]
    
    padding = 0
    if use_padding:
        window_size = win.shape[3]
        padding = window_size // 2

    #same 1D filter for both axes    
    out = F.conv2d(input, win, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out

#convenient alias
apply_gaussian_filter = apply_1Dfilter



#TODO: use this in the initialization of class FilterX, so it can be used on 
# forward with an image (LoG, Gaussian, etc)
def load_filter(kernel, kernel_size=3, in_channels=3, out_channels=3, 
                stride=1, padding=True, groups=3, dim: int =2, 
                requires_grad=False):
    '''
        Loads a kernel's coefficients into a Conv layer that 
            can be used to convolve an image with, by default, 
            for depthwise convolution
        Can use nn.Conv1d, nn.Conv2d or nn.Conv3d, depending on
            the dimension set in dim (1,2,3)
        #From Pytorch Conv2D:
            https://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
            When `groups == in_channels` and `out_channels == K * in_channels`,
            where `K` is a positive integer, this operation is also termed in
            literature as depthwise convolution.
             At groups= :attr:`in_channels`, each input channel is convolved with
             its own set of filters, of size:
             :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.
    '''

    '''#TODO: check if this is necessary, probably not
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    '''

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel_conv_w(kernel, in_channels)
    assert(len(kernel.shape)==4 and kernel.shape[0]==in_channels)

    if padding:
        pad = compute_padding(kernel_size)
    else:
        pad = 0
    
    # create filter as convolutional layer
    if dim == 1:
        conv = nn.Conv1d
    elif dim == 2:
        conv = nn.Conv2d
    elif dim == 3:
        conv = nn.Conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported for convolution. \
            Received {}.'.format(dim)
        )

    filter = conv(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding, 
                        groups=groups, bias=False)
    filter.weight.data = kernel
    filter.weight.requires_grad = requires_grad
    return filter


def compute_padding(kernel_size):
    '''
        Computes padding tuple. For square kernels, pad can be an
         int, else, a tuple with an element for each dimension
    '''
    # 4 or 6 ints:  (padding_left, padding_right, padding_top, padding_bottom)
    if isinstance(kernel_size, int):
        return kernel_size//2
    elif isinstance(kernel_size, list):
        computed = [k // 2 for k in kernel_size]

        out_padding = []

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]
            # for even kernels we need to do asymetric padding
            if kernel_size[i] % 2 == 0:
                padding = computed_tmp - 1
            else:
                padding = computed_tmp
            out_padding.append(padding)
            out_padding.append(computed_tmp)
        return out_padding

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect', 
             dim: int =2,
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.
    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    #if not len(input.shape) == 4:
        #raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         #.format(input.shape))

    #if not len(kernel.shape) == 3:
        #raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         #.format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel) 
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)
    b, c, hp, wp = input_pad.shape

    tmp_kernel = tmp_kernel.expand(c, -1, -1, -1)

    # convolve the tensor with the kernel.
    if dim == 1:
        conv = F.conv1d
    elif dim == 2:
        conv = F.conv2d
        #TODO: this needs a review, the final sizes don't match with .view(b, c, h, w), (they are larger).
            # using .view(b, c, -1, w) results in an output, but it's 3 times larger than it should be
        '''
        # if kernel_numel > 81 this is a faster algo
        kernel_numel: int = height * width #kernel_numel = torch.numel(tmp_kernel[-1:])
        if kernel_numel > 81:
            return conv(input_pad.reshape(b * c, 1, hp, wp), tmp_kernel, padding=0, stride=1).view(b, c, h, w)
        '''
    elif dim == 3:
        conv = F.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
        )

    return conv(input_pad, tmp_kernel, groups=c, padding=0, stride=1)


#TODO: make one class to receive any arbitrary kernel and others that are specific (like gaussian, etc)
#class FilterX(nn.Module):
  #def __init__(self, ..., kernel_type, dim: int=2):
      #r"""
      #Args:
          #argument: ...
      #"""
      #super(filterXd, self).__init__()
      #Here receive an pre-made kernel of any type, load as tensor or as
      #convXd layer (class or functional)
      # self.filter = load_filter(kernel=kernel, kernel_size=kernel_size, 
                #in_channels=image_channels, out_channels=image_channels, stride=stride, 
                #padding=pad, groups=image_channels)
  #def forward:
      #This would apply the filter that was initialized
    


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, padding=True, 
                image_channels=3, include_pad=True, filter_type=None):
        super(FilterLow, self).__init__()
        
        if padding:
            pad = compute_padding(kernel_size)
        else:
            pad = 0
        
        if filter_type == 'gaussian':
            sigma = get_kernel_sigma(kernel_size)
            kernel = get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma)
            self.filter = load_filter(kernel=kernel, kernel_size=kernel_size, 
                    in_channels=image_channels, stride=stride, padding=pad)
        #elif filter_type == '': #TODO... box? (the same as average) What else?
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, 
                    padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=9, stride=1, include_pad=True, 
            image_channels=3, normalize=True, filter_type=None, kernel=None):
        super(FilterHigh, self).__init__()
        
        # if is standard freq. separator, will use the same LPF to remove LF from image
        if filter_type=='gaussian' or filter_type=='average':
            self.type = 'separator'
            self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, 
                image_channels=image_channels, include_pad=include_pad, filter_type=filter_type)
        # otherwise, can use any independent filter
        else: #load any other filter for the high pass
            self.type = 'independent'
            #kernel and kernel_size should be provided. Options for edge detectors:
            # In both dimensions: get_log_kernel, get_laplacian_kernel_3x3 
            # and get_sobel_kernel
            # Single dimension: get_prewitt_kernel_3x3, get_scharr_kernel_3x3 
            # get_gradient_kernel_3x3 
            if include_pad:
                pad = compute_padding(kernel_size)
            else:
                pad = 0
            self.filter_low = load_filter(kernel=kernel, kernel_size=kernel_size, 
                in_channels=image_channels, out_channels=image_channels, stride=stride, 
                padding=pad, groups=image_channels)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.type == 'separator':
            if self.recursions > 1:
                for i in range(self.recursions - 1):
                    img = self.filter_low(img)
            img = img - self.filter_low(img)
        elif self.type == 'independent':
            img = self.filter_low(img)
        if self.normalize:
            return denorm(img)
        else:
            return img

#TODO: check how similar getting the gradient with get_gradient_kernel_3x3 is from the alternative displacing the image
#ref from TF: https://github.com/tensorflow/tensorflow/blob/4386a6640c9fb65503750c37714971031f3dc1fd/tensorflow/python/ops/image_ops_impl.py#L3423
def get_image_gradients(image):
    """Returns image gradients (dy, dx) for each color channel.
    Both output tensors have the same shape as the input: [b, c, h, w]. 
    Places the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y). 
    That means that dy will always have zeros in the last row,
    and dx will always have zeros in the last column.

    This can be used to implement the anisotropic 2-D version of the 
    Total Variation formula:
        https://en.wikipedia.org/wiki/Total_variation_denoising
    (anisotropic is using l1, isotropic is using l2 norm)
    
    Arguments:
        image: Tensor with shape [b, c, h, w].
    Returns:
        Pair of tensors (dy, dx) holding the vertical and horizontal image
        gradients (1-step finite difference).  
    Raises:
      ValueError: If `image` is not a 3D image or 4D tensor.
    """
    
    image_shape = image.shape
      
    if len(image_shape) == 3:
        # The input is a single image with shape [height, width, channels].
        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        dx = image[:, 1:, :] - image[:, :-1, :] #pixel_dif2, f_v_1-f_v_2
        dy = image[1:, :, :] - image[:-1, :, :] #pixel_dif1, f_h_1-f_h_2

    elif len(image_shape) == 4:    
        # Return tensors with same size as original image
        #adds one pixel pad to the right and removes one pixel from the left
        right = F.pad(image, [0, 1, 0, 0])[..., :, 1:]
        #adds one pixel pad to the bottom and removes one pixel from the top
        bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :] 

        #right and bottom have the same dimensions as image
        dx, dy = right - image, bottom - image 
        
        #this is required because otherwise results in the last column and row having 
        # the original pixels from the image
        dx[:, :, :, -1] = 0 # dx will always have zeros in the last column, right-left
        dy[:, :, -1, :] = 0 # dy will always have zeros in the last row,    bottom-top
    else:
      raise ValueError(
          'image_gradients expects a 3D [h, w, c] or 4D tensor '
          '[batch_size, c, h, w], not %s.', image_shape)

    return dy, dx


def get_4dim_image_gradients(image):
    # Return tensors with same size as original image
    # Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
    right = F.pad(image, [0, 1, 0, 0])[..., :, 1:] #adds one pixel pad to the right and removes one pixel from the left
    bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :] #adds one pixel pad to the bottom and removes one pixel from the top
    botright = F.pad(image, [0, 1, 0, 1])[..., 1:, 1:] #displaces in diagonal direction

    dx, dy = right - image, bottom - image #right and bottom have the same dimensions as image
    dn, dp = botright - image, right - bottom
    #dp is positive diagonal (bottom left to top right)
    #dn is negative diagonal (top left to bottom right)
    
    #this is required because otherwise results in the last column and row having 
    # the original pixels from the image
    dx[:, :, :, -1] = 0 # dx will always have zeros in the last column, right-left
    dy[:, :, -1, :] = 0 # dy will always have zeros in the last row,    bottom-top
    dp[:, :, -1, :] = 0 # dp will always have zeros in the last row

    return dy, dx, dp, dn

#TODO: #https://towardsdatascience.com/implement-canny-edge-detection-from-scratch-with-pytorch-a1cccfa58bed
#TODO: https://link.springer.com/article/10.1007/s11220-020-00281-8
def grad_orientation(grad_y, grad_x):
    go = torch.atan(grad_y / grad_x)
    go = go * (360 / np.pi) + 180 # convert to degree
    go = torch.round(go / 45) * 45  # keep a split by 45
    return go
################################################################################################################################################################################################
# from dataops.colors import *
# codes/dataops/colors.py

'''
Functions for color operations on tensors.
If needed, there are more conversions that can be used:
https://github.com/kornia/kornia/tree/master/kornia/color
https://github.com/R08UST/Color_Conversion_pytorch/blob/master/differentiable_color_conversion/basic_op.py
'''


import torch
import math
import cv2

def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    out: torch.Tensor = image.flip(-3) #https://github.com/pytorch/pytorch/issues/229
    #out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out

def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)

def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out

def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)

def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    #gray = rgb_to_yuv(input,consts='y')
    return gray

def bgr_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    input_rgb = bgr_to_rgb(input)
    gray: torch.Tensor = rgb_to_grayscale(input_rgb)
    #gray = rgb_to_yuv(input_rgb,consts='y')
    return gray

def grayscale_to_rgb(input: torch.Tensor) -> torch.Tensor:
    #repeat the gray image to the three channels
    rgb: torch.Tensor = input.repeat(3, *[1] * (input.dim() - 1))
    return rgb

def grayscale_to_bgr(input: torch.Tensor) -> torch.Tensor:
    return grayscale_to_rgb(input)

def rgb_to_ycbcr(input: torch.Tensor, consts='yuv'):
    return rgb_to_yuv(input, consts == 'ycbcr')

def rgb_to_yuv(input: torch.Tensor, consts='yuv'):
    """Converts one or more images from RGB to YUV.
    Outputs a tensor of the same shape as the `input` image tensor, containing the YUV
    value of the pixels.
    The output is only well defined if the value in images are in [0,1].
    Y′CbCr is often confused with the YUV color space, and typically the terms YCbCr 
    and YUV are used interchangeably, leading to some confusion. The main difference 
    is that YUV is analog and YCbCr is digital: https://en.wikipedia.org/wiki/YCbCr
    Args:
      input: 2-D or higher rank. Image data to convert. Last dimension must be
        size 3. (Could add additional channels, ie, AlphaRGB = AlphaYUV)
      consts: YUV constant parameters to use. BT.601 or BT.709. Could add YCbCr
        https://en.wikipedia.org/wiki/YUV
    Returns:
      images: images tensor with the same shape as `input`.
    """
    
    #channels = input.shape[0]
    
    if consts == 'BT.709': # HDTV YUV
        Wr = 0.2126
        Wb = 0.0722
        Wg = 1 - Wr - Wb #0.7152
        Uc = 0.539
        Vc = 0.635
        delta: float = 0.5 #128 if image range in [0,255]
    elif consts == 'ycbcr': # Alt. BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Uc = 0.564 #(b-y) #cb
        Vc = 0.713 #(r-y) #cr
        delta: float = .5 #128 if image range in [0,255]
    elif consts == 'yuvK': # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Ur = -0.147
        Ug = -0.289
        Ub = 0.436
        Vr = 0.615
        Vg = -0.515
        Vb = -0.100
        #delta: float = 0.0
    elif consts == 'y': #returns only Y channel, same as rgb_to_grayscale()
        #Note: torchvision uses ITU-R 601-2: Wr = 0.2989, Wg = 0.5870, Wb = 0.1140
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
    else: # Default to 'BT.601', SDTV YUV
        Wr = 0.299
        Wb = 0.114
        Wg = 1 - Wr - Wb #0.587
        Uc = 0.493 #0.492
        Vc = 0.877
        delta: float = 0.5 #128 if image range in [0,255]

    r: torch.Tensor = input[..., 0, :, :]
    g: torch.Tensor = input[..., 1, :, :]
    b: torch.Tensor = input[..., 2, :, :]
    #TODO
    #r, g, b = torch.chunk(input, chunks=3, dim=-3) #Alt. Which one is faster? Appear to be the same. Differentiable? Kornia uses both in different places

    if consts == 'y':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        #(0.2989 * input[0] + 0.5870 * input[1] + 0.1140 * input[2]).to(img.dtype)
        return y
    elif consts == 'yuvK':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = Ur * r + Ug * g + Ub * b
        v: torch.Tensor = Vr * r + Vg * g + Vb * b
    else: #if consts == 'ycbcr' or consts == 'yuv' or consts == 'BT.709':
        y: torch.Tensor = Wr * r + Wg * g + Wb * b
        u: torch.Tensor = (b - y) * Uc + delta #cb
        v: torch.Tensor = (r - y) * Vc + delta #cr

    if consts == 'uv': #returns only UV channels
        return torch.stack((u, v), -2) 
    else:
        return torch.stack((y, u, v), -3) 

def ycbcr_to_rgb(input: torch.Tensor):
    return yuv_to_rgb(input, consts == 'ycbcr')

def yuv_to_rgb(input: torch.Tensor, consts='yuv') -> torch.Tensor:
    if consts == 'yuvK': # Alt. yuv from Kornia YUV values: https://github.com/kornia/kornia/blob/master/kornia/color/yuv.py
        Wr = 1.14 #1.402
        Wb = 2.029 #1.772
        Wgu = 0.396 #.344136
        Wgv = 0.581 #.714136
        delta: float = 0.0
    elif consts == 'yuv' or consts == 'ycbcr': # BT.601 from Kornia YCbCr values, from JPEG conversion
        Wr = 1.403 #1.402
        Wb = 1.773 #1.772
        Wgu = .344 #.344136
        Wgv = .714 #.714136
        delta: float = .5 #128 if image range in [0,255]
    
    #Note: https://github.com/R08UST/Color_Conversion_pytorch/blob/75150c5fbfb283ae3adb85c565aab729105bbb66/differentiable_color_conversion/basic_op.py#L65 has u and v flipped
    y: torch.Tensor = input[..., 0, :, :]
    u: torch.Tensor = input[..., 1, :, :] #cb
    v: torch.Tensor = input[..., 2, :, :] #cr
    #TODO
    #y, u, v = torch.chunk(input, chunks=3, dim=-3) #Alt. Which one is faster? Appear to be the same. Differentiable? Kornia uses both in different places

    u_shifted: torch.Tensor = u - delta #cb
    v_shifted: torch.Tensor = v - delta #cr

    r: torch.Tensor = y + Wr * v_shifted
    g: torch.Tensor = y - Wgv * v_shifted - Wgu * u_shifted
    b: torch.Tensor = y + Wb * u_shifted
    return torch.stack((r, g, b), -3) 

#Not tested:
def rgb2srgb(imgs):
    return torch.where(imgs<=0.04045,imgs/12.92,torch.pow((imgs+0.055)/1.055,2.4))

#Not tested:
def srgb2rgb(imgs):
    return torch.where(imgs<=0.0031308,imgs*12.92,1.055*torch.pow((imgs),1/2.4)-0.055)
################################################################################################################################################################################################
# from dataops.common import norm, denorm
# dnorm shon copy pasted

def norm(x): 
    #Normalize (z-norm) from [0,1] range to [-1,1]
    out = (x - 0.5) * 2.0
    if isinstance(x, torch.Tensor):
        return out.clamp(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, -1, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")





















################################################################################################################################################################################################

################################################################################################################################################################################################
# https://github.com/victorca25/BasicSR/blob/dev2/codes/models/modules/loss.py

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)
    

# Define GAN loss: [vanilla | lsgan | wgan-gp | srpgan/nsgan | hinge]
# https://tuatini.me/creating-and-shipping-deep-learning-models-into-production/
class GANLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'srpgan' or self.gan_type == 'nsgan':
            self.loss = nn.BCELoss()
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val) #torch.ones_like(d_sr_out)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val) #torch.zeros_like(d_sr_out)

    def forward(self, input, target_is_real, is_disc = None):
        if self.gan_type == 'hinge': #TODO: test
            if is_disc:
                input = -input if target_is_real else input
                return self.loss(1 + input).mean()
            else:
                return (-input).mean()
        else:
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)
            return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


class HFENLoss(nn.Module): # Edge loss with pre_smooth
    """Calculates high frequency error norm (HFEN) between target and 
     prediction used to quantify the quality of reconstruction of edges 
     and fine features. 
     
     Uses a rotationally symmetric LoG (Laplacian of Gaussian) filter to 
     capture edges. The original filter kernel is of size 15×15 pixels, 
     and has a standard deviation of 1.5 pixels.
     ks = 2 * int(truncate * sigma + 0.5) + 1, so use truncate=4.5
     
     HFEN is computed as the norm of the result obtained by LoG filtering the 
     difference between the reconstructed and reference images.

    [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning, 2011
        https://ieeexplore.ieee.org/document/5617283
    [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016
        https://www.hindawi.com/journals/cmmm/2016/7571934/
    
    Parameters
    ----------
    img1 : torch.Tensor or torch.autograd.Variable
        Predicted image
    img2 : torch.Tensor or torch.autograd.Variable
        Target image
    norm: if true, follows [2], who define a normalized version of HFEN.
        If using RelativeL1 criterion, it's already normalized. 
    """
    def __init__(self, loss_f=None, kernel='log', kernel_size=15, sigma = 2.5, norm = False): #1.4 ~ 1.5
        super(HFENLoss, self).__init__()
        # can use different criteria
        self.criterion = loss_f
        self.norm = norm
        #can use different kernels like DoG instead:
        if kernel == 'dog':
            kernel = get_dog_kernel(kernel_size, sigma)
        else:
            kernel = get_log_kernel(kernel_size, sigma)
        self.filter = load_filter(kernel=kernel, kernel_size=kernel_size)

    def forward(self, img1, img2):
        self.filter.to(img1.device)
        # HFEN loss
        log1 = self.filter(img1)
        log2 = self.filter(img2)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= img2.norm()
        return hfen_loss


class TVLoss(nn.Module):
    def __init__(self, tv_type='tv', p = 1):
        super(TVLoss, self).__init__()
        assert p in [1, 2]
        self.p = p
        self.tv_type = tv_type

    def forward(self, x):
        img_shape = x.shape
        if len(img_shape) == 3 or len(img_shape) == 4:
            if self.tv_type == 'dtv':
                dy, dx, dp, dn  = get_4dim_image_gradients(x)

                if len(dy.shape) == 3:
                    # Sum for all axis. (None is an alias for all axis.)
                    reduce_axes = None
                    batch_size = 1
                elif len(dy.shape) == 4:
                    # Only sum for the last 3 axis.
                    # This results in a 1-D tensor with the total variation for each image.
                    reduce_axes = (-3, -2, -1)
                    batch_size = x.size()[0]
                #Compute the element-wise magnitude of a vector array
                # Calculates the TV for each image in the batch
                # Calculate the total variation by taking the absolute value of the
                # pixel-differences and summing over the appropriate axis.
                if self.p == 1:
                    loss = (dy.abs().sum(dim=reduce_axes) + dx.abs().sum(dim=reduce_axes) + dp.abs().sum(dim=reduce_axes) + dn.abs().sum(dim=reduce_axes)) # Calculates the TV loss for each image in the batch
                elif self.p == 2:
                    loss = torch.pow(dy,2).sum(dim=reduce_axes) + torch.pow(dx,2).sum(dim=reduce_axes) + torch.pow(dp,2).sum(dim=reduce_axes) + torch.pow(dn,2).sum(dim=reduce_axes)
                # calculate the scalar loss-value for tv loss
                loss = loss.sum()/(2.0*batch_size) # averages the TV loss all the images in the batch (note: the division is not in TF version, only the sum reduction)
                return loss
            else: #'tv'
                dy, dx  = get_image_gradients(x)

                if len(dy.shape) == 3:
                    # Sum for all axis. (None is an alias for all axis.)
                    reduce_axes = None
                    batch_size = 1
                elif len(dy.shape) == 4:
                    # Only sum for the last 3 axis.
                    # This results in a 1-D tensor with the total variation for each image.
                    reduce_axes = (-3, -2, -1)
                    batch_size = x.size()[0]
                #Compute the element-wise magnitude of a vector array
                # Calculates the TV for each image in the batch
                # Calculate the total variation by taking the absolute value of the
                # pixel-differences and summing over the appropriate axis.
                if self.p == 1:
                    loss = dy.abs().sum(dim=reduce_axes) + dx.abs().sum(dim=reduce_axes)
                elif self.p == 2:
                    loss = torch.pow(dy,2).sum(dim=reduce_axes) + torch.pow(dx,2).sum(dim=reduce_axes)
                # calculate the scalar loss-value for tv loss
                loss = loss.sum()/batch_size # averages the TV loss all the images in the batch (note: the division is not in TF version, only the sum reduction)
                return loss
        else:
            raise ValueError("Expected input tensor to be of ndim 3 or 4, but got " + str(len(img_shape)))
    

class GradientLoss(nn.Module):
    def __init__(self, loss_f = None, reduction='mean', gradientdir='2d'): #2d or 4d
        super(GradientLoss, self).__init__()
        self.criterion = loss_f
        self.gradientdir = gradientdir
    
    def forward(self, input, target):
        if self.gradientdir == '4d':
            inputdy, inputdx, inputdp, inputdn = get_4dim_image_gradients(input)
            targetdy, targetdx, targetdp, targetdn = get_4dim_image_gradients(target) 
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy) + \
                    self.criterion(inputdp, targetdp) + self.criterion(inputdn, targetdn))/4
        else: #'2d'
            inputdy, inputdx = get_image_gradients(input)
            targetdy, targetdx = get_image_gradients(target) 
            return (self.criterion(inputdx, targetdx) + self.criterion(inputdy, targetdy))/2


class ElasticLoss(nn.Module):
    def __init__(self, a=0.2, reduction='mean'): #a=0.5 default
        super(ElasticLoss, self).__init__()
        self.alpha = torch.FloatTensor([a, 1 - a]) #.to('cuda:0')
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        for i in range(len(input)):
            l2 = F.mse_loss(input[i].squeeze(), target.squeeze(), reduction=self.reduction).mul(self.alpha[0])
            l1 = F.l1_loss(input[i].squeeze(), target.squeeze(), reduction=self.reduction).mul(self.alpha[1])
            loss = l1 + l2

        return loss


#TODO: change to RelativeNorm and set criterion as an input argument, could be any basic criterion
class RelativeL1(nn.Module):
    '''
    Comparing to the regular L1, introducing the division by |c|+epsilon 
    better models the human vision system’s sensitivity to variations
    in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
    denominator)
    '''
    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):
        base = target + self.eps
        return self.criterion(input/base, target/base)


class L1CosineSim(nn.Module):
    '''
    https://github.com/dmarnerides/hdr-expandnet/blob/master/train.py
    Can be used to replace L1 pixel loss, but includes a cosine similarity term 
    to ensure color correctness of the RGB vectors of each pixel.
    lambda is a constant factor that adjusts the contribution of the cosine similarity term
    It provides improved color stability, especially for low luminance values, which
    are frequent in HDR images, since slight variations in any of theRGB components of these 
    low values do not contribute much totheL1loss, but they may however cause noticeable 
    color shifts. More in the paper: https://arxiv.org/pdf/1803.02266.pdf
    '''
    def __init__(self, loss_lambda=5, reduction='mean'):
        super(L1CosineSim, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


class ClipL1(nn.Module):
    '''
    Clip L1 loss
    From: https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution/
    ClipL1 Loss combines Clip function and L1 loss. self.clip_min sets the 
    gradients of well-trained pixels to zeros and clip_max works as a noise filter.
    data range [0, 255]: (clip_min=0.0, clip_max=10.0), 
    for [0,1] set clip_min to 1/255=0.003921.
    '''
    def __init__(self, clip_min=0.0, clip_max=10.0):
        super(ClipL1, self).__init__()
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, sr, hr):
        loss = torch.mean(torch.clamp(torch.abs(sr-hr), self.clip_min, self.clip_max))
        return loss


# Frequency loss 
# https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/fft.py
class FFTloss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean'):
        super(FFTloss, self).__init__()
        self.criterion = loss_f(reduction=reduction)

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).to(img1.device)
        return self.criterion(torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))


class OFLoss(torch.nn.Module):
    '''
    Overflow loss
    Only use if the image range is in [0,1]. (This solves the SPL brightness problem
    and can be useful in other cases as well)
    https://github.com/lj1995-computer-vision/Trident-Dehazing-Network/blob/master/loss/brelu.py
    '''
    def __init__(self):
        super(OFLoss, self).__init__()

    def forward(self, img1):
        img_clamp = img1.clamp(0,1)
        b,c,h,w = img1.shape
        return torch.log((img1 - img_clamp).abs() + 1).sum()/b/c/h/w


#TODO: testing
# Color loss 
class ColorLoss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean', ds_f=None):
        super(ColorLoss, self).__init__()
        self.ds_f = ds_f
        self.criterion = loss_f

    def forward(self, input, target):
        input_uv = rgb_to_yuv(self.ds_f(input), consts='uv')
        target_uv = rgb_to_yuv(self.ds_f(target), consts='uv')
        return self.criterion(input_uv, target_uv)

#TODO: testing
# Averaging Downscale loss 
class AverageLoss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean', ds_f=None):
        super(AverageLoss, self).__init__()
        self.ds_f = ds_f
        self.criterion = loss_f

    def forward(self, input, target):
        input_uv = rgb_to_yuv(self.ds_f(input), consts='uv')
        target_uv = rgb_to_yuv(self.ds_f(target), consts='uv')
        return self.criterion(input_uv, target_uv)




########################
# Spatial Profile Loss
########################

class GPLoss(nn.Module):
    '''
    https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/
    Gradient Profile (GP) loss
    The image gradients in each channel can easily be computed 
    by simple 1-pixel shifted image differences from itself. 
    '''
    def __init__(self, trace=False, spl_denorm=False):
        super(GPLoss, self).__init__()
        self.spl_denorm = spl_denorm
        if trace == True: # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
        else: # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()

    def __call__(self, input, reference):
        ## Use "spl_denorm" when reading a [-1,1] input, but you want to compute the loss over a [0,1] range
        # Note: only rgb_to_yuv() requires image in the [0,1], so this denorm is optional, depending on the net
        if self.spl_denorm == True:
            input = denorm(input)
            reference = denorm(reference)
        input_h, input_v = get_image_gradients(input)
        ref_h, ref_v = get_image_gradients(reference)

        trace_v = self.trace(input_v,ref_v)
        trace_h = self.trace(input_h,ref_h)
        return trace_v + trace_h

class CPLoss(nn.Module):
    '''
    Color Profile (CP) loss
    '''
    def __init__(self, rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.spl_denorm = spl_denorm
        self.yuv_denorm = yuv_denorm
        
        if trace == True: # Alternate behavior: use the complete calculation with SPL_ComputeWithTrace()
            self.trace = SPL_ComputeWithTrace()
            self.trace_YUV = SPL_ComputeWithTrace()
        else: # Default behavior: use the more efficient SPLoss()
            self.trace = SPLoss()
            self.trace_YUV = SPLoss()

    def __call__(self, input, reference):
        ## Use "spl_denorm" when reading a [-1,1] input, but you want to compute the loss over a [0,1] range
        # self.spl_denorm=False when your inputs and outputs are in [0,1] range already
        # Note: only rgb_to_yuv() requires image in the [0,1], so this denorm is optional, depending on the net
        if self.spl_denorm:
            input = denorm(input)
            reference = denorm(reference)
        total_loss= 0
        if self.rgb:
            total_loss += self.trace(input,reference)
        if self.yuv:
            # rgb_to_yuv() needs images in [0,1] range to work
            if not self.spl_denorm and self.yuv_denorm:
                input = denorm(input)
                reference = denorm(reference)
            input_yuv = rgb_to_yuv(input)
            reference_yuv = rgb_to_yuv(reference)
            total_loss += self.trace(input_yuv,reference_yuv)
        if self.yuvgrad:
            input_h, input_v = get_image_gradients(input_yuv)
            ref_h, ref_v = get_image_gradients(reference_yuv)

            total_loss +=  self.trace(input_v,ref_v)
            total_loss +=  self.trace(input_h,ref_h)

        return total_loss

## Spatial Profile Loss (SPL) with trace
class SPL_ComputeWithTrace(nn.Module):
    """
    Spatial Profile Loss (SPL)
    Both loss versions equate to the cosine similarity of rows/columns. 
    'SPL_ComputeWithTrace()' uses the trace (sum over the diagonal) of matrix multiplication 
    of L2-normalized input/target rows/columns.
    Slow implementation of the trace loss using the same formula as stated in the paper. 
    In principle, we compute the loss between a source and target image by considering such 
    pattern differences along the image x and y-directions. Considering a row or a column 
    spatial profile of an image as a vector, we can compute the similarity between them in 
    this induced vector space. Formally, this similarity is measured over each image channel ’c’.
    The first term computes similarity among row profiles and the second among column profiles 
    of an image pair (x, y) of size H ×W. These image pixels profiles are L2-normalized to 
    have a normalized cosine similarity loss.
    """
    def __init__(self,weight = [1.,1.,1.]): # The variable 'weight' was originally intended to weigh color channels differently. In our experiments, we found that an equal weight between all channels gives the best results. As such, this variable is a leftover from that time and can be removed.
        super(SPL_ComputeWithTrace, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                a += torch.trace(torch.matmul(F.normalize(input[i,j,:,:],p=2,dim=1),torch.t(F.normalize(reference[i,j,:,:],p=2,dim=1))))/input.shape[2]*self.weight[j]
                b += torch.trace(torch.matmul(torch.t(F.normalize(input[i,j,:,:],p=2,dim=0)),F.normalize(reference[i,j,:,:],p=2,dim=0)))/input.shape[3]*self.weight[j]
        a = -torch.sum(a)/input.shape[0]
        b = -torch.sum(b)/input.shape[0]
        return a+b

## Spatial Profile Loss (SPL) without trace, prefered
class SPLoss(nn.Module):
    ''' 
    Spatial Profile Loss (SPL)
    'SPLoss()' L2-normalizes the rows/columns, performs piece-wise multiplication 
    of the two tensors and then sums along the corresponding axes. This variant 
    needs less operations since it can be performed batchwise.
    Note: SPLoss() makes image results too bright, when using images in the [0,1] 
    range and no activation as output of the Generator.
    SPL_ComputeWithTrace() does not have this problem, but results are very blurry. 
    Adding the Overflow Loss fixes this problem.
    '''
    def __init__(self):
        super(SPLoss, self).__init__()
        #self.weight = weight

    def __call__(self, input, reference):
        a = torch.sum(torch.sum(F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2),dim=2, keepdim=True))
        b = torch.sum(torch.sum(F.normalize(input, p=2, dim=3) * F.normalize(reference, p=2, dim=3),dim=3, keepdim=True))
        return -(a + b) / (input.size(2) * input.size(0))





########################
# Contextual Loss
########################

DIS_TYPES = ['cosine', 'l1', 'l2']

class Contextual_Loss(nn.Module):
    '''
    Contextual loss for unaligned images (https://arxiv.org/abs/1803.02077)

    https://github.com/roimehrez/contextualLoss
    https://github.com/S-aiueo32/contextual_loss_pytorch
    https://github.com/z-bingo/Contextual-Loss-PyTorch

    layers_weights: is a dict, e.g., {'conv_1_1': 1.0, 'conv_3_2': 1.0}
    crop_quarter: boolean
    '''
    def __init__(self, layers_weights, crop_quarter=False, max_1d_size=100, 
            distance_type: str = 'cosine', b=1.0, band_width=0.5, 
            use_vgg: bool = True, net: str = 'vgg19', calc_type: str =  'regular'):
        super(Contextual_Loss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert distance_type in DIS_TYPES,\
            f'select a distance type from {DIS_TYPES}.'

        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
        
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.band_width = band_width #self.h = h, #sigma
        
        if use_vgg:
            self.vgg_model = VGG_Model(listen_list=listen_list, net=net)

        if calc_type == 'bilateral':
            self.calculate_loss = self.bilateral_CX_Loss
        elif calc_type == 'symetric':
            self.calculate_loss = self.symetric_CX_Loss
        else: #if calc_type == 'regular':
            self.calculate_loss = self.calculate_CX_Loss

    def forward(self, images, gt):
        device = images.device
        
        if hasattr(self, 'vgg_model'):
            assert images.shape[1] == 3 and gt.shape[1] == 3,\
                'VGG model takes 3 channel images.'
            
            loss = 0
            vgg_images = self.vgg_model(images)
            vgg_images = {k: v.clone().to(device) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_model(gt)
            vgg_gt = {k: v.to(device) for k, v in vgg_gt.items()}

            for key in self.layers_weights.keys():
                if self.crop_quarter:
                    vgg_images[key] = self._crop_quarters(vgg_images[key])
                    vgg_gt[key] = self._crop_quarters(vgg_gt[key])

                N, C, H, W = vgg_images[key].size()
                if H*W > self.max_1d_size**2:
                    vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                    vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

                loss_t = self.calculate_loss(vgg_images[key], vgg_gt[key])
                loss += loss_t * self.layers_weights[key]
                # del vgg_images[key], vgg_gt[key]
        #TODO: without VGG it runs, but results are not looking right
        else:
            if self.crop_quarter:
                images = self._crop_quarters(images)
                gt = self._crop_quarters(gt)

            N, C, H, W = images.size()
            if H*W > self.max_1d_size**2:
                images = self._random_pooling(images, output_1d_size=self.max_1d_size)
                gt = self._random_pooling(gt, output_1d_size=self.max_1d_size)

            loss = self.calculate_loss(images, gt)
        return loss

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        device=tensor.device
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.clamp(indices.min(), tensor.shape[-1]-1) #max = indices.max()-1
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = indices.to(device)

        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = Contextual_Loss._random_sampling(feats[0], output_1d_size**2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature_tensor):
        N, fC, fH, fW = feature_tensor.size()
        quarters_list = []
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature_tensor[..., round(fH / 2):, 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False
            )
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        # prepare feature before calculating cosine distance
        # mean shifting by channel-wise mean of `y`.
        mean_T = T_features.mean(dim=(0, 2, 3), keepdim=True)        
        I_features = I_features - mean_T
        T_features = T_features - mean_T

        # L2 channelwise normalization
        I_features = F.normalize(I_features, p=2, dim=1)
        T_features = F.normalize(T_features, p=2, dim=1)
        
        N, C, H, W = I_features.size()
        cosine_dist = []
        # work seperatly for each example in dim 1
        for i in range(N):
            # channel-wise vectorization
            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous() # 1CHW --> 11CP, with P=H*W
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            #cosine_dist.append(dist) # back to 1CHW
            #TODO: temporary hack to workaround AMP bug:
            cosine_dist.append(dist.to(torch.float32)) # back to 1CHW
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)

        return cosine_dist

    #compute_relative_distance
    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon) # Eq 2
        return relative_dist

    def symetric_CX_Loss(self, I_features, T_features):
        loss = (self.calculate_CX_Loss(T_features, I_features) + self.calculate_CX_Loss(I_features, T_features)) / 2
        return loss #score

    def bilateral_CX_Loss(self, I_features, T_features, weight_sp: float = 0.1):
        def compute_meshgrid(shape):
            N, C, H, W = shape
            rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
            cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

            feature_grid = torch.meshgrid(rows, cols)
            feature_grid = torch.stack(feature_grid).unsqueeze(0)
            feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

            return feature_grid

        # spatial loss
        grid = compute_meshgrid(I_features.shape).to(T_features.device)
        raw_distance = Contextual_Loss._create_using_L2(grid, grid) # calculate raw distance
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width) # Eq(3)
        cx_sp = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # feature loss
        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width) # Eq(3)
        cx_feat = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # combined loss
        cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp
        k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
        cx = k_max_NC.mean(dim=1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss

    def calculate_CX_Loss(self, I_features, T_features):
        device = I_features.device
        T_features = T_features.to(device)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(torch.isinf(I_features)) == torch.numel(I_features):
            print(I_features)
            raise ValueError('NaN or Inf in I_features')
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
                torch.isinf(T_features)) == torch.numel(T_features):
            print(T_features)
            raise ValueError('NaN or Inf in T_features')

        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(
                torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError('NaN or Inf in raw_distance')

        # normalizing the distances
        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(
                torch.isinf(relative_distance)) == torch.numel(relative_distance):
            print(relative_distance)
            raise ValueError('NaN or Inf in relative_distance')
        del raw_distance

        #compute_sim()
        # where h>0 is a band-width parameter
        exp_distance = torch.exp((self.b - relative_distance) / self.band_width) # Eq(3)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance
        
        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance
        
        #contextual_loss()
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0] # Eq(1)
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS)) # Eq(5)
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')
        return CX_loss

  



































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

        # original loss
        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.generator_loss = config.GENERATOR_LOSS

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

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

    def process(self, images, edges, masks, mosaic_size=None):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        if(mosaic_size != None):
          # resize image with random size. (256 currently hardcoded)
          #mosaic_size = int(random.triangular(int(min(256*0.01, 256*0.01)), int(min(256*0.2, 256*0.2)), int(min(256*0.0625, 256*0.0625))))
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




            # original generator loss
            # generator adversarial loss
            gen_input_fake = outputs
            
            if 'DEFAULT_GAN' in self.generator_loss:
              gen_fake, _ = self.discriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
              gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
              gen_loss += gen_gan_loss

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
          dis_real, _ = self.discriminator(DiffAugment(dis_input_real, policy=policy))                    # in: [rgb(3)]
          dis_fake, _ = self.discriminator(DiffAugment(dis_input_fake, policy=policy))                    # in: [rgb(3)]
          dis_real_loss = self.adversarial_loss(dis_real, True, True)
          dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
          dis_loss += (dis_real_loss + dis_fake_loss) / 2




          # original generator loss
          # generator adversarial loss
          gen_input_fake = outputs
          
          if 'DEFAULT_GAN' in self.generator_loss:
            gen_fake, _ = self.discriminator(DiffAugment(gen_input_fake, policy=policy))                  # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss

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

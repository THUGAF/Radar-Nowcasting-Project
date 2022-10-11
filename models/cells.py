import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    r"""ConvLSTM Cell without peephole connection.
        Arguments:
            channels (int): number of input channels
            filters (int): number of convolutional kernels
            kernel_size (int, tuple): size of convolutional kernels
    """

    def __init__(self, channels, filters, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.filters = filters
        self.conv = nn.Sequential(
            nn.Conv2d(channels + filters, filters * 4, kernel_size, padding=padding),
            nn.BatchNorm2d(filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x, h, c):
        # x: 4D tensor (B, C, H, W)
        batch_size, channels, height, width = x.size()
        # Initialize h and c with torch.zeros
        if h is None:
            h = torch.zeros(size=(batch_size, self.filters, height, width), device=x.device)
        if c is None:
            c = torch.zeros(size=(batch_size, self.filters, height, width), device=x.device)
        # forward process
        i, f, g, o = torch.split(self.conv(torch.cat([x, h], dim=1)), self.filters, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


class ConvCell(nn.Module):
    r"""A convolutional cell for 5D (S, B, C, H, W) input. The ConvCell consists of 2 parts, 
        the ResNet bottleneck and the SENet module (optional). 
        
        Arguments:
            channels (int): Number of input channels
            filters (int): Number of convolutional kernels
            kernel_size (int or tuple): Size of convolutional kernels
            stride (int or tuple): Stride of the convolution
            padding (int or tuple): Padding of the convolution
    """

    def __init__(self, channels, filters, kernel_size, stride, padding): 
        super(ConvCell, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class DeconvCell(nn.Module):
    r"""A transpose convolutional cell for 5D (S, B, C, H, W) input. The DeconvCell consists of 2 parts, 
        the ResNet bottleneck and the SENet module (optional). 
        
        Arguments:
            channels (int): Number of input channels
            filters (int): Number of convolutional kernels
            kernel_size (int or tuple): Size of convolutional kernels
            stride (int or tuple): Stride of the convolution
            padding (int or tuple): Padding of the convolution
            output_padding (int or tuple): Additional size added to one side of each dimension in the output shape
    """
    
    def __init__(self, channels, filters, kernel_size, stride, padding, output_padding, up_size):
        super(DeconvCell, self).__init__()

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(size=up_size),
            nn.ConvTranspose2d(channels, filters, kernel_size=kernel_size, stride=1, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.up(x)

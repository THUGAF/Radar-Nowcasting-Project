import torch
import torch.nn as nn
from .cells import ConvLSTMCell, ConvCell, DeconvCell


class Encoder(nn.Module):
    r"""A 3-layer ConvLSTM Encoder for 5D (S, B, C, H, W) input. Each layer of the encoder includes a ConvLSTM layer 
        and a convolutional layer.
        Arguments:
            channels (tuple): Number of channels of each encoding layer
            kernels (tuple): Size of convolutional kernels of each encoding layer. Default: (3, 3, 3)
            strides tuple): Stride size of the convolution of each encoding layer. Default: (2, 2, 2)
            paddings (int or tuple): Additional size added to one side of each dimension in the convolution
    """

    def __init__(self, channels, kernels=(3,)*6, strides=(2,)*3, paddings=(1,)*6):
        super(Encoder, self).__init__()
        
        c1, c2, c3 = channels
        k11, k12, k21, k22, k31, k32 = kernels
        s1, s2, s3 = strides
        p11, p12, p21, p22, p31, p32 = paddings

        self.down1 = ConvCell(1, c1, k11, s1, p11)
        self.rnn1 = ConvLSTMCell(c1, c1, k12, p12)
        self.down2 = ConvCell(c1, c2, k21, s2, p21)
        self.rnn2 = ConvLSTMCell(c2, c2, k22, p22)
        self.down3 = ConvCell(c2, c3, k31, s3, p31)
        self.rnn3 = ConvLSTMCell(c3, c3, k32, p32)

    def forward(self, input_):
        
        # input_: 5D tensor (S, B, C, H, W)
        in_len = input_.size(0)
        h_list = [None, None, None]
        c_list = [None, None, None]
        
        for i in range(in_len):
            y = self.down1(input_[i])
            h_list[0], c_list[0] = self.rnn1(y, h_list[0], c_list[0])
            y = self.down2(h_list[0])
            h_list[1], c_list[1] = self.rnn2(y, h_list[1], c_list[1])
            y = self.down3(h_list[1])
            h_list[2], c_list[2] = self.rnn3(y, h_list[2], c_list[2])
        
        return h_list, c_list


class Forecaster(nn.Module):
    r"""A 3-layer ConvLSTM Encoder for 5D (S, B, C, H, W) input. Each layer of the encoder includes a ConvLSTM layer 
        and a convolutional layer.
        Arguments:
            channels (tuple): Number of channels of each encoding layer
            kernels (tuple): Size of convolutional kernels of each encoding layer. Default: (3, 3, 3)
            strides (tuple): Stride size of the convolution of each encoding layer. Default: (2, 2, 2)
            output_padding (int or tuple): Additional size added to one side of each dimension in the output shape.
                Default: (0, 0, 0)
    """

    def __init__(self, out_len, up_size, channels, kernels=(3,)*6, strides=(2,)*3, paddings=(1,)*6, out_paddings=(0,)*3):
        super(Forecaster, self).__init__()

        c1, c2, c3 = channels
        k11, k12, k21, k22, k31, k32 = kernels
        s1, s2, s3 = strides
        p11, p12, p21, p22, p31, p32 = paddings
        o1, o2, o3 = out_paddings
        u1, u2, u3 = up_size
        self.out_len = out_len

        self.rnn3 = ConvLSTMCell(c3, c3, k32, p32)
        self.up3 = DeconvCell(c3, c2, k31, s3, p31, o3, u3)
        self.rnn2 = ConvLSTMCell(c2, c2, k22, p22)
        self.up2 = DeconvCell(c2, c1, k21, s2, p21, o2, u2)
        self.rnn1 = ConvLSTMCell(c1, c1, k12, p12)
        self.up1 = DeconvCell(c1, 1, k11, s1, p11, o1, u1)

    
    def forward(self, h_list, c_list):
        
        # initialize input_ with torch.zeros
        input_ = torch.zeros(size=(self.out_len,) + tuple(h_list[-1].size()), device=h_list[-1].device)
        output = []
        
        for i in range(self.out_len):
            y = input_[i]
            h_list[2], c_list[2] = self.rnn3(y, h_list[2], c_list[2])
            y = self.up3(h_list[2])
            h_list[1], c_list[1] = self.rnn2(y, h_list[1], c_list[1])
            y = self.up2(h_list[1])
            h_list[0], c_list[0] = self.rnn1(y, h_list[0], c_list[0])
            y = self.up1(h_list[0])
            output.append(y)
    
        # output: 5D tensor (S_out, B, C, H, W)
        output = torch.stack(output, dim=0)
        return output


class EncoderForecaster(nn.Module):
    def __init__(self, out_len, up_size, channels, kernels, strides, paddings, out_paddings):
        super(EncoderForecaster, self).__init__()
        self.encoder = Encoder(channels, kernels, strides, paddings)
        self.forecaster = Forecaster(out_len, up_size, channels, kernels, strides, paddings, out_paddings)
    
    def forward(self, input_):
        states, cells = self.encoder(input_)
        output = self.forecaster(states, cells)
        self.states = states
        self.cells = cells
        return output


if __name__ == "__main__":
    x = torch.randn(5, 2, 3, 101, 101)
    ef = EncoderForecaster(10, (101, 51, 26), (16, 32, 64),
                           (3, 3, 3, 3, 3, 3), (2, 2, 2), (1,)*6, (0, 0, 0))
    y = ef(x)
    print(y.size())

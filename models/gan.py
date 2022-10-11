import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, channels, kernels, strides, paddings):
        super(Discriminator, self).__init__()

        c1, c2, c3, c4, c5, c6 = channels
        k1, k2, k3, k4, k5, k6 = kernels
        s1, s2, s3, s4, s5, s6 = strides
        p1, p2, p3, p4, p5, p6 = paddings

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, c1, k1, stride=s1, padding=p1),
            nn.BatchNorm3d(c1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(c1, c2, k2, stride=s2, padding=p2),
            nn.BatchNorm3d(c2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(c2, c3, k3, stride=s3, padding=p3),
            nn.BatchNorm3d(c3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(c3, c4, k4, stride=s4, padding=p4),
            nn.BatchNorm3d(c4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(c4, c5, k5, stride=s5, padding=p5),
            nn.BatchNorm3d(c5),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(c5, c6, k6, stride=s6, padding=p6),
            nn.Sigmoid()
        )
    
    def forward(self, input_):
        seq_len, batch_size, _, _, _ = input_.size()
        input_ = input_.permute(1, 2, 0, 3, 4)
        output = self.conv1(input_)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.squeeze()
        return output
    

class DCGAN(nn.Module):
    def __init__(self, generator, channels, kernels, strides, paddings):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = Discriminator(channels, kernels, strides, paddings)
    
    def forward(self, input_):
        return self.generator(input_)


if __name__ == "__main__":
    x = torch.randn(10, 4, 1, 101, 101)
    d = Discriminator((16, 32, 64, 128, 256, 1), (3, 3, 3, 3, 3, 3), 
                      ((1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 1, 1)), 
                      (1, 1, 1, 1, 0, 0))
    y = d(x)
    print(y.size())

EF_channels = (8, 32, 64)
EF_kernels = (3, 3, 3, 3, 3, 3)
EF_strides = (2, 2, 2)
EF_paddings = (1, 1, 1, 1, 1, 1)
EF_out_paddings = (0, 0, 0)

GAN_channels = (16, 32, 64, 128, 256, 1)
GAN_kernels = (3, 3, 3, 3, 3, 3)
GAN_strides = (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 1, 1)
GAN_paddings = (1, 1, 1, 1, 0, 0)

UP_size = (101, 51, 26)

MULTI_GPU = [0, 1, 2, 3, 4, 5, 6, 7]

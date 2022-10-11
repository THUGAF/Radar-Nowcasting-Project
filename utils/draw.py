import os
import matplotlib.pyplot as plt
import imageio
import numpy as np


def draw_loss(train_loss, val_loss, path='results/loss/loss'):
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax1 = plt.subplot(111)
    _draw_loss_fig(ax1, train_loss, val_loss, 'Loss')
    plt.savefig(path + '.png')
    plt.savefig(path + '.eps', format='eps')
    plt.close()


def draw_gan_loss(train_loss_D, train_loss_G, val_loss_D, val_loss_G, path='results/loss/loss_gan'):
    fig = plt.figure(figsize=(16, 6), dpi=150)
    ax1 = plt.subplot(121)
    _draw_loss_fig(ax1, train_loss_D, val_loss_D, 'Loss D')
    ax2 = plt.subplot(122)
    _draw_loss_fig(ax2, train_loss_G, val_loss_G, 'Loss G')
    plt.savefig(path + '.png')
    plt.savefig(path + '.eps', format='eps')
    plt.close()


def _draw_loss_fig(ax, train_loss, val_loss, title):
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b')
    ax.plot(range(1, len(val_loss) + 1), val_loss, 'r')
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.legend(['train loss', 'val loss'])


def draw_rain(input_, pred, truth, root='results/images', stage='train'):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, stage)):
        os.mkdir(os.path.join(root, stage))
    _draw_rain_figs(input_, root, stage, 'input')
    _draw_rain_figs(pred, root, stage, 'pred')
    _draw_rain_figs(truth, root, stage, 'truth')


def _draw_rain_figs(tensor, root, stage, data_type):
    path = os.path.join(root, stage, data_type)
    if not os.path.exists(path):
        os.mkdir(path)

    image_list = []
    for i in range(tensor.size(0)):
        # minus represents the time before current moment
        if data_type == 'input':
            str_min = str(6 * (i - tensor.size(0) + 1))
        else:
            str_min = str(6 * (i + 1))
        title = '{} {} {} min'.format(stage, data_type, str_min)
        file_path ='{}/{}.png'.format(path, str_min)
        _draw_rain_fig(tensor[i, 0, 0].cpu(), file_path, title)
        image_list.append(imageio.imread(file_path))
    # make gif
    imageio.mimsave('{}/{}.gif'.format(path, data_type), image_list, 'GIF', duration=1.0)


def _draw_rain_fig(tensor_slice, file_path, title):
    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = plt.subplot(111)
    ax.imshow(tensor_slice, cmap='jet', clim=[0, 1])
    ax.set_title(title)
    ax.axis('off')
    plt.savefig(file_path)
    plt.savefig(file_path.replace('.png', '.eps'), format='eps')
    plt.close()

"""
if __name__ == "__main__":
    train_loss = list(np.loadtxt("results/loss/train_loss.txt"))
    val_loss = list(np.loadtxt("results/loss/val_loss.txt"))
    draw_loss(train_loss, val_loss)
"""

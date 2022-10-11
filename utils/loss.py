import torch
import torch.nn
import torch.nn.functional as F


def cal_loss_D(model, pred, truth):
    fake = model(pred)
    real = model(truth)
    label = torch.full(fake.size(), 1.0).to(pred.device)
    loss_D_fake = F.binary_cross_entropy(fake, label * 0.0)
    loss_D_real = F.binary_cross_entropy(real, label * 1.0)
    loss_D = loss_D_fake + loss_D_real
    
    return loss_D


def cal_loss_G(model, pred, truth, l=0.02):
    fake = model(pred)
    label = torch.full(fake.size(), 1.0).to(pred.device)
    # mixed loss function for generator: adversial loss + image loss
    loss_G = l * F.binary_cross_entropy(fake, label * 1.0) + \
        F.mse_loss(pred, truth) + F.l1_loss(pred, truth)
    
    return loss_G


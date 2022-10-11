import os
import argparse
import time
import shutil
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.convrnn import EncoderForecaster
from utils.ssim import ssim, ssim_each
from utils.evaluation import evaluate, evaluate_each
from utils.dataloader import MyDataset, load_data
from utils.draw import draw_rain, draw_loss
from utils.config import *


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='DL-Nowcasting')

parser.add_argument('-tr', '--train', action='store_true')
parser.add_argument('-te', '--test', action='store_true')
parser.add_argument('-b', '--batch-size', type=int, default=5)
parser.add_argument('-d', '--data', type=int, default=14)
parser.add_argument('-i', '--in-len', type=int, default=5)
parser.add_argument('-o', '--out-len', type=int, default=10)
parser.add_argument('-l', '--lr', type=float, default=5e-4)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-f', '--freq', type=int, default=10)
parser.add_argument('-g', '--gpu', type=int, default=0)
parser.add_argument('-c', '--cpu', action='store_true')
parser.add_argument('-mg', '--multi-gpu', action='store_true')
parser.add_argument('-od', '--output-dir', type=str, default='results')
parser.add_argument('--start-epoch', type=int, default=0)

args = parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', bestname='bestmodel.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    min_loss = checkpoint['min_loss']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return model, optimizer, start_epoch, min_loss


def train_epoch(model, train_loader, epoch, epochs, optimizer, freq, stage='train'):
    model.train()
    loss_batches = []
    
    # train the model on a batch of train set
    for i, data in enumerate(train_loader):      
        t = time.time()
        
        # load data from CPU to target DEVICE and split input and ground truth
        data = data.transpose(1, 0).to(DEVICE)      # (S, B, C, H, W)
        input_ = data[:args.in_len]
        truth = data[args.in_len:, :, 1: 2]
        
        # forecasting
        pred = model(input_)
        
        # back propogation
        optimizer.zero_grad()
        loss = F.mse_loss(pred, truth) + F.l1_loss(pred, truth)
        loss.backward()
        optimizer.step()

        loss_batches.append(loss.item())

        # print training performance on a batch
        if (i + 1) % freq == 0:
            print('Epoch: [{}][{}]    Batch: [{}][{}]    Time: {:6.4f}    Loss: {:6.4f}'\
                .format(epoch + 1, epochs, i + 1, len(train_loader), time.time() - t, loss.item()))  
        t = time.time()

        # draw a training image
        if i == len(train_loader) - 1:
            draw_rain(input_, pred.detach(), truth, root=args.output_dir + '/images', stage=stage)
    
    return loss_batches


def val_epoch(model, val_loader, epoch, epochs, freq, stage='val'):
    model.eval()
    loss_batches = []

    # validate the model on a batch of val set
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            t = time.time()
            
            # load data from CPU to target DEVICE and split input and ground truth
            data = data.transpose(1, 0).to(DEVICE)
            input_ = data[:args.in_len]
            truth = data[args.in_len:, :, 1: 2]
            
            # forecasting
            pred = model(input_)
            
            loss = F.mse_loss(pred, truth) + F.l1_loss(pred, truth)
            loss_batches.append(loss.item())

            # print validation performance on a batch
            if (i + 1) % freq == 0:
                print('Epoch: [{}][{}]    Batch: [{}][{}]    Time: {:6.4f}    Loss: {:6.4f}'\
                    .format(epoch + 1, epochs, i + 1, len(val_loader), time.time() - t, loss.item()))  
            t = time.time()

            # draw a validating image
            if i == len(val_loader) - 1:
                draw_rain(input_, pred.detach(), truth, root=args.output_dir + '/images', stage=stage)

    return loss_batches


def train(model, train_loader, val_loader, epochs, optimizer, freq, has_checkpoint=False):
    train_loss = []
    val_loss = []

    # load checkpoint
    if has_checkpoint is True:
        model, optimizer, start_epoch, min_loss = load_checkpoint(model, optimizer)
    else:
        start_epoch = 0
        min_loss = 999999
    
    # train, validate and save model parameters
    for epoch in range(start_epoch, epochs):

        # train for each epoch
        t = time.time()
        print('\nEpoch: [{}][{}]'.format(epoch + 1, epochs))
        print('\n[Train]')
        train_loss_batches = train_epoch(model, train_loader, epoch, epochs, optimizer, freq)
        print('Epoch: [{}][{}]    Time: {:6.4f}    Loss: {:6.4f}'
              .format(epoch + 1, epochs, time.time() - t, train_loss_batches[-1]))
        
        # validate for each epoch
        t = time.time()
        print('\n[Validate]')
        val_loss_batches = val_epoch(model, val_loader, epoch, epochs, freq)
        print('Epoch: [{}][{}]    Time: {:6.4f}    Loss: {:6.4f}'
              .format(epoch + 1, epochs, time.time() - t, np.mean(val_loss_batches)))

        train_loss.append(train_loss_batches[-1])
        val_loss.append(np.mean(val_loss_batches))

        # save checkpoint and find the best model
        is_best = np.mean(val_loss_batches) < min_loss
        min_loss = min(min(val_loss), min_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'min_loss': min_loss
        }, is_best)

        # plot curves
        draw_loss(train_loss, val_loss, path=args.output_dir + '/loss/loss')

        np.savetxt(args.output_dir + '/loss/train_loss.txt', train_loss)
        np.savetxt(args.output_dir + '/loss/val_loss.txt', val_loss)


def test(model, test_loader, freq=10, stage='test'):
    model.eval()
    loss_batches = []
    eval_batches = []
    ssim_batches = []
    mse_batches = []
    
    eval_each_batches = []
    ssim_each_batches = []
    mse_each_batches = []
    ssim_each_arr = np.zeros(args.out_len)
    mse_each_arr = np.zeros(args.out_len)

    tt = time.time()
    print('\n[Test]')

    # test the model on a batch of test set
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            t = time.time()
            
            data = data.transpose(1, 0).to(DEVICE)
            input_ = data[:args.in_len]
            truth = data[args.in_len:, :, 1: 2]

            # forecasting
            pred = model(input_)

            loss = F.mse_loss(pred, truth) + F.l1_loss(pred, truth)
            
            eval_ = evaluate(pred.detach(), truth.detach())
            ssim_ = ssim(pred.detach(), truth.detach())
            mse = F.mse_loss(pred.detach(), truth.detach())

            eval_each = evaluate_each(pred.detach(), truth.detach())
            for j in range(args.out_len):
                ssim_each_arr[j] = ssim_each(pred[j].detach(), truth[j].detach()).item()
                mse_each_arr[j] = F.mse_loss(pred[j].detach(), truth[j].detach()).item()

            loss_batches.append(loss.item())
            eval_batches.append(eval_)
            ssim_batches.append(ssim_.item())
            mse_batches.append(mse.item())

            eval_each_batches.append(eval_each)
            ssim_each_batches.append(ssim_each_arr)
            mse_each_batches.append(mse_each_arr)

            # print validation performance on a batch
            if (i + 1) % freq == 0:
                print('Batch: [{}][{}]    Time: {:6.4f}    Loss'
                      .format(i + 1, len(test_loader), time.time() - t, loss.item()))
            t = time.time()

            # draw a test image
            if i == len(test_loader) - 1:
                draw_rain(input_, pred.detach(), truth, root=args.output_dir + '/images', stage=stage)
            
    test_loss = np.mean(loss_batches)
    test_eval = np.mean(eval_batches, axis=0)
    test_ssim = np.mean(ssim_batches)
    test_mse = np.mean(mse_batches)
    test_eval_each = np.mean(eval_each_batches, axis=0)
    test_ssim_each = np.mean(ssim_each_batches, axis=0)
    test_mse_each = np.mean(mse_each_batches, axis=0)

    np.savetxt(args.output_dir + '/loss/test_loss_ssim_mse.txt', np.array([test_loss, test_ssim, test_mse]))
    np.savetxt(args.output_dir + '/loss/test_eval.txt', np.array(test_eval))
    np.savetxt(args.output_dir + '/loss/test_eval_each.txt', np.array(test_eval_each))
    np.savetxt(args.output_dir + '/loss/test_ssim_each.txt', np.array(test_ssim_each))
    np.savetxt(args.output_dir + '/loss/test_mse_each.txt', np.array(test_mse_each))

    print('Time: {:6.4f}    Loss: {:6.4f}'.format(time.time() - tt, test_loss))


def sample(model, sample_loader, freq=10, stage='sample'):
    model.eval()

    # test the model on a batch of test set
    with torch.no_grad():
        for i, data in enumerate(sample_loader):
            data = data.transpose(1, 0).to(DEVICE)
            input_ = data[:args.in_len]
            truth = data[args.in_len:, :, 1: 2]

            pred = model(input_)
 
            # draw a sample image
            draw_rain(input_, pred.detach(), truth, root=args.output_dir +
                      '/images', stage='{}_{}'.format(stage, str(i)))
    
    print('\nsample done')


def main_worker():
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device('cpu')
    if args.cpu:
        DEVICE = torch.device('cpu')

    # create output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(os.path.join(args.output_dir, 'images'))
        os.mkdir(os.path.join(args.output_dir, 'loss'))

    # load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = load_data([0, args.data], batch_size=args.batch_size)
    sample_loader = DataLoader(MyDataset(997, 1000, is_sample=True), batch_size=1)

    # train, validate and test
    model = EncoderForecaster(args.out_len, UP_size, EF_channels, EF_kernels, EF_strides, EF_paddings, EF_out_paddings).to(DEVICE)
    print('Parameters of EF:', count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    has_checkpoint = False if args.start_epoch == 0 else True

    if args.train:
        train(model, train_loader, val_loader, args.epochs, optimizer, freq=args.freq, has_checkpoint=has_checkpoint)
    bestmodel = load_checkpoint(model, optimizer, filename='bestmodel.pth.tar')[0]
    if args.test:
        test(bestmodel, test_loader, freq=args.freq)

    print('\nsample loader len:', len(sample_loader))
    sample(bestmodel, sample_loader, freq=args.freq)


if __name__ == "__main__":
    main_worker()

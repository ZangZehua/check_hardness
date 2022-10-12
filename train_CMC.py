"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import csv
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import datetime

import tensorboard_logger as tb_logger

from torchvision import transforms
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import ImageFolderInstance


def args_parse():
    parser = argparse.ArgumentParser('argument for pre-training')
    pretrain_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d%H%M%S"))

    # base
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')

    # argsimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--random_epoch', type=int, default=None, help='no help')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='stl10', choices=['imagenet100', 'imagenet', "tiny",
                                                                         "stl10", "cifar10", "cifar100"])

    # specify folder
    parser.add_argument('--data_folder', type=str, default="/home/hujie/zdata/data/", help='path to data')
    parser.add_argument('--model_path', type=str, default="models_pt", help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs_pt", help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # mixed precision setting
    parser.add_argument('--args_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    parser.add_argument('--max_1', type=float, default=0.2, help='max loss of augmented data')
    parser.add_argument('--max_2', type=float, default=1.0, help='max loss of augmented data')
    parser.add_argument('--candidate_number', type=int, default=50, help='the number of candidate augmented data')
    parser.add_argument('--check_method', type=int, default=1, choices=[1, 2], help='1-max_loss=ori_loss*(1+max_1),'
                                                                                    '2-max_loss=ori_loss+max_2')

    args = parser.parse_args()

    args.data_base_path = args.data_folder
    if args.dataset == "stl10":
        save_path_base = "saved/STL-10_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "STL-10")
    elif args.dataset == "cifar10":
        save_path_base = "saved/CIFAR-10_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "CIFAR-10")
    elif args.dataset == "cifar100":
        save_path_base = "saved/CIFAR-100_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "CIFAR-100")
    elif args.dataset == "tiny":
        save_path_base = "saved/Tiny_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "tiny-imagenet-200")
    elif args.dataset == "imagenet":
        save_path_base = "saved/IMAGENET_" + pretrain_time
        args.data_folder = os.path.join(args.data_folder, "imagenet")
    else:
        raise FileNotFoundError

    if args.check_method == 1:
        save_path_base += "_1_" + str(args.max_1)
    elif args.check_method == 2:
        save_path_base += "_2_" + str(args.max_2)

    args.model_path = os.path.join(save_path_base, args.model_path)
    args.tb_path = os.path.join(save_path_base, args.tb_path)

    if args.random_epoch is not None:
        args.resume = os.path.join(args.data_base_path, os.path.join("cmc_models_" + args.dataset, "ckpt_epoch_" + str(args.random_epoch) + ".pth"))

    # create save folder
    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)
        args.save_path_base = save_path_base

    if args.dataset == 'imagenet':
        if 'alexnet' not in args.model:
            args.crop_low = 0.08

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.isdir(args.tb_path):
        os.makedirs(args.tb_path)

    if not os.path.isdir(args.data_folder):
        raise ValueError('data path not exist: {}'.format(args.data_folder))

    args.measure_path = os.path.join(save_path_base, "measures.csv")

    return args


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder, 'unlabeled')

    if args.view == 'Lab':
        mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
        std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
        color_transfer = RGB2Lab()
    elif args.view == 'YCbCr':
        mean = [116.151, 121.080, 132.342]
        std = [109.500, 111.855, 111.964]
        color_transfer = RGB2YCbCr()
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        # transforms.RandomHorizontalFlip(),
        color_transfer,
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def generate_batch(inputs, indexes, candidate_number, check_method, max_1, max_2,
                   model, contrast, criterion_l, criterion_ab, measure_csv):
    model.eval()
    outputs = torch.Tensor([]).cuda()
    for image, index in zip(inputs, indexes):
        measure_list = []
        max_measure = -999
        max_image = image
        # get ori_loss
        with torch.no_grad():
            feat_l, feat_ab = model(max_image.unsqueeze(0))
            out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, index)
            base_measure = criterion_l(out_l) + criterion_ab(out_ab)
            if check_method == 1:
                max_range = base_measure.item() * (1 + max_1)
            elif check_method == 2:
                max_range = base_measure.item() + max_2
            measure_list.append(base_measure.item())
        # generate candidate_number image
        for _ in range(candidate_number):
            temp_image = transforms.RandomResizedCrop(224, scale=(0.2, 1))(image).unsqueeze(0)
            temp_image = transforms.RandomHorizontalFlip()(temp_image)
            with torch.no_grad():
                feat_l, feat_ab = model(temp_image)
                out_l, out_ab = contrast.get_out_l_ab(feat_l, feat_ab, index)
                measure = criterion_l(out_l) + criterion_ab(out_ab)
                if max_measure < measure.item() <= max_range:
                    max_image = temp_image
                    max_measure = measure.item()
            measure_list.append(measure.item())
        print(max_measure)
        outputs = torch.cat((outputs, max_image), dim=0)
        measure_csv.writerow(measure_list)
    model.train()
    return outputs


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, args, measure_csv):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda()
            inputs = inputs.cuda()

        inputs = generate_batch(inputs, index, args.candidate_number, args.check_method, args.max_1, args.max_2,
                                model, contrast, criterion_l, criterion_ab, measure_csv)
        # ===================forward=====================
        feat_l, feat_ab = model(inputs)
        out_l, out_ab = contrast(feat_l, feat_ab, index)

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, lprobs=l_prob_meter,
                abprobs=ab_prob_meter))
            sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():
    # parse the args
    args = args_parse()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    measure_file = open(args.measure_path, 'a+', encoding='utf-8', newline='')
    measure_csv = csv.writer(measure_file)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_path, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args, measure_csv)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('l_loss', l_loss, epoch)
        logger.log_value('l_prob', l_prob, epoch)
        logger.log_value('ab_loss', ab_loss, epoch)
        logger.log_value('ab_prob', ab_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'args': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import config
from model import resnet50,se_resnet68
#from resnet_revquantized import resnet_revquantized


from lars import LARS

best_acc1 = 0

config = config.config


def main():
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)

    print(config)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = min(args.lr * (0.1 ** (epoch >= 30)) * (0.1 ** (epoch >= 60)) * (0.1 ** (epoch >= 80))  *  \
         ( 0.2 *  min(epoch+1, 5)) , 1.6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main_worker(gpu, ngpus_per_node, config):
    global best_acc1
    config.gpu = gpu

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
    # create model
    if config.fixgamma:
        if config.pretrained:
            print("=> using pre-trained model '{}'".format(config.arch))
            model = models.__dict__[config.arch](pretrained=True)
        else:
            print("=> Running Fix Gamma BN Version of ResNet50.")
            model = resnet50(warp=config.warp,half=config.half)

    else:
        if config.pretrained:
            print("=> using pre-trained model '{}'".format(config.arch))
            model = models.__dict__[config.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(config.arch))
            model = models.__dict__[config.arch]()
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int(config.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if config.arch.startswith('alexnet') or config.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if not config.bce:
        criterion = nn.CrossEntropyLoss().cuda(config.gpu)
    else:
        criterion = nn.BCEWithLogitsLoss().cuda(config.gpu)

    def add_weight_decay(net, l2_value, skip_list=()):
        decay, no_decay = [], []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue # frozen weight
            if (len(param.shape) == 1 or name.endswith("gamma"))  or name in skip_list:
                no_decay.append(param)
            else: decay.append(param)
            return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

    print([name for name,p in model.named_parameters() if 'gamma' in name or 'bias' in name])
    #params = add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.SGD([{'params': [p for name,p in model.named_parameters() if 'gamma' in name or 'bias' in name], 'weight_decay':0.0 },
                                {'params': [p for name,p in model.named_parameters() if 'gamma' not in name and 'bias' not in name]}],
                                config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay
                                )

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(config.data, 'train')
    valdir = os.path.join(config.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.458, 0.4076],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.4,saturation=0.4,brightness=0.4),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize( ((config.imgsize//32)+1)*32  ),
            transforms.CenterCrop(config.imgsize),
            transforms.ToTensor(),
            normalize,
        ]))

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1 if config.noeval else config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=val_sampler) # sampler=val_sampler

    if config.evaluate:
        validate(val_loader, model, criterion, config)
        return

    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, config)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    if config.noeval:
        model.train(mode=False)
        #model.eval()
    else:
        model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.half :
                input = input.half()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            if args.bce:
                n = target.size()[0]
                targetm = torch.FloatTensor(n, 1000).fill_(0)
                for j in range(n):
                    targetm[j,target[j]]=1
                if args.half:
                    targetm.half()
                targetm = targetm.cuda(args.gpu, non_blocking=True)
            
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            if args.bce:
                loss = criterion(output, targetm)
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, args):
    print(args.stepper)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.half:
            input = input.half()
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        if args.bce:
            n = target.size()[0]
            targetm = torch.FloatTensor(n, 1000).fill_(0)
            for j in range(n):
                targetm[j,target[j]]=1
            # Random Label DropOut
            if args.half:
                targetm.half()
            targetm = targetm.cuda(args.gpu, non_blocking=True)
        
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        if args.bce:
            loss = criterion(output, targetm)
        else:
            loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        #if i % args.stepper == 0:
        #    optimizer.zero_grad()
        loss.sum().backward()
        if (i + 1) % args.stepper == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5))
    optimizer.step()
    optimizer.zero_grad()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    print("Start Training")
    main()

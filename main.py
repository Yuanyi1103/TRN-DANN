import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import network
import loss as los
from torch.nn.utils import clip_grad_norm
from torchvision import models

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import datasets_video


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    check_rootfolders()

    categories, args.train_source_list, args.train_target_list, args.val_list, args.root_path_source, args.root_path_target, prefix = datasets_video.return_dataset(args.dataset, args.modality)
    num_class = len(categories)


    args.store_name = '_'.join(['TRN', args.dataset, args.modality, args.arch, args.consensus_type, 'segment%d'% args.num_segments])
    print('storing name: ' + args.store_name)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    
    source_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path_source, args.train_source_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    target_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path_target, args.train_target_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path_target, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(source_loader, target_loader, model, criterion, optimizer, epoch, log_training, num_class)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(source_loader, target_loader, model, criterion, optimizer, epoch, log, num_class):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_a = AverageMeter()  #adversarial loss
    losses_c = AverageMeter()  #classification loss
    losses = AverageMeter()  #final loss
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)

    end = time.time()
    data_loader = enumerate(zip(source_loader, target_loader))

    #start_steps = epoch * len(source_loader)
    #total_steps = args.epochs * len(source_loader)
    #base_network = net_config["name"](**net_config["params"])
    #base_network = network.ResNetFc(models.resnet101, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000)
    #base_network = base_network.cuda() 

    #random_layer = network.RandomLayer([base_network.output_num(), num_class], 1024)
    #random_layer.cuda()
    ad_net = network.AdversarialNetwork(224*224*9, 96)
   
    ad_net = ad_net.cuda()
    #print(ad_net)
    
    for i, ((source_data, source_label), (target_data, target_label)) in data_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        source_label = source_label.cuda(async=True)
        target_label = target_label.cuda(async=True)

        #======calculate loss function=====#
        #1. calculate classification loss losses_c
        
        source_input_var = torch.autograd.Variable(source_data)
        source_target_var = torch.autograd.Variable(source_label)

        output = model(source_input_var)
        loss_c = criterion(output, source_target_var)
        losses_c.update(loss_c.data[0], source_data.size(0))
        loss = loss_c

        #2. calculate the adversarial loss loss_a
        #loss_a = 0
        #base_network.train(True)
        #base_network.train(True)
        #ad_net.train(True)
         
        if i % len_train_source == 0:
            iter_source = iter(source_loader)
        if i % len_train_target == 0:
            iter_target = iter(target_loader)

        target_input_var = torch.autograd.Variable(target_data)
        target_target_var = torch.autograd.Variable(target_label)
        
        #if i % len_train_source == 0:
        #    iter_source = iter(source_input_var)
        #if i % len_train_target == 0:
        #    iter_target = iter(target_input_var)

        source_label_frame = source_label.unsqueeze(1).repeat(1,args.num_segments).view(-1)
        #print(source_label_frame)
        target_label_frame = target_label.unsqueeze(1).repeat(1,args.num_segments).view(-1)
  
     
        inputs_source, labels_source = iter_source.next()

        inputs_target, labels_target = iter_target.next()
        #inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
 
        features_source = source_input_var.type(torch.cuda.FloatTensor)
        features_target = target_input_var.type(torch.cuda.FloatTensor)
        #features_source, outputs_source = base_network(torch.autograd.Variable(inputs_source))
        #features_target, outputs_target = base_network(torch.autograd.Variable(inputs_target))
        features = torch.cat((features_source, features_target), dim=0)
        features = features.view(features.size(0), -1)
        #print(features)
        #features = features.unsqueeze(1).repeat(1,args.num_segments).view(-1)
        #print(source_label_frame)
        
        loss_a = los.DANN(features, ad_net)
        loss += loss_a

        #losses_a.update(loss_a.data[0], pred_domain.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, source_label, topk=(1,5))
        
        top1.update(prec1[0], source_input_var.size(0))
        top5.update(prec5[0], source_input_var.size(0))        
        #top1.update(prec1[0], source_data.size(0))
        #top5.update(prec5[0], source_data.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        losses.update(loss.data[0])
        loss.backward()
        #loss_c.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(source_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()



def validate(val_loader, model, criterion, iter, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),'%s/%s_best.pth.tar' % (args.root_model, args.store_name))

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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()

import argparse
import os, sys
import time

import random
import json
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from dataset.get_dataset import get_datasets, TransformUnlabeled_WS

from dataset.handlers import COCO2014_mask_handler, VOC2012_mask_handler, NUS_WIDE_mask_handler

from models.MLDResnet import resnet50_ml_decoder
from models.Resnet import create_model

from utils.logger import setup_logger
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from utils.helper import clean_state_dict, function_mAP, get_raw_dict, ModelEma, add_weight_decay
from utils.losses import AsymmetricLoss

np.set_printoptions(suppress=True, precision=4)

MASK_HANDLER_DICT = {'voc': VOC2012_mask_handler, 'coco': COCO2014_mask_handler, 'nus': NUS_WIDE_mask_handler}

NUM_CLASS = {'voc': 20, 'coco': 80, 'nus': 81}

def parser_args():
    parser = argparse.ArgumentParser(description='Main')

    # data
    parser.add_argument('--dataset_name', default='coco', choices=['voc', 'coco', 'nus'], 
                        help='dataset name')
    parser.add_argument('--dataset_dir',  default='./data', metavar='DIR', 
                        help='dir of all datasets')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--output', default='./outputs', metavar='DIR', 
                        help='path to output folder')

    # train
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--warmup_batch_size', default=32, type=int,
                        help='batch size for warmup')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,metavar='W', 
                        help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=400, type=int, metavar='N', 
                        help='print frequency (default: 10)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--save_PR', action='store_true', default=False,
                        help='on/off PR')
    parser.add_argument('--optim', default='adamw', type=str,
                        help='optimizer used')
    parser.add_argument('--warmup_epochs', default=12, type=int,
                        help='the number of epochs for warmup')
    parser.add_argument('--lb_ratio', default=0.05, type=float,
                        help='the ratio of lb:(lb+ub)')
    parser.add_argument('--loss_lb', default='asl', type=str, 
                        help='used_loss for lb')
    parser.add_argument('--loss_ub', default='asl', type=str, 
                        help='used_loss for ub')
    parser.add_argument('--cutout', default=0.5, type=float, 
                        help='cutout factor')

    parser.add_argument('--init_pos_per', default=1.0, type=float, 
                        help='init pos_per')
    parser.add_argument('--init_neg_per', default=1.0, type=float, 
                        help='init neg_per')




    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    # model
    parser.add_argument('--net', default='resnet50', type=str, choices=['resnet50', 'mlder'],
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--is_data_parallel', action='store_true', default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--resume', default=None, type=str, 
                        help='path to latest checkpoint (default: none)')


    args = parser.parse_args()

    if args.lb_ratio == 0.05:
        args.lb_bs = 4
        args.ub_bs = 60
    elif args.lb_ratio == 0.1:
        args.lb_bs = 8
        args.ub_bs = 56
    elif args.lb_ratio == 0.15:
        args.lb_bs = 12
        args.ub_bs = 52
    elif args.lb_ratio == 0.2:
        args.lb_bs = 16
        args.ub_bs = 48

    if args.dataset_name == 'voc':
        args.lb_bs = int(args.lb_bs / 2)
        args.ub_bs = int(args.ub_bs / 2)

    args.output = args.net + '_outputs'
    args.resume = '%s_outputs/%s/%s/%s/warmup_%s_%s/warmup_model.pth.tar'%(args.net, args.dataset_name, args.img_size, args.lb_ratio, args.loss_lb, args.warmup_epochs)
    args.n_classes = NUM_CLASS[args.dataset_name]
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name) 
    
    args.output = os.path.join(args.output, args.dataset_name, '%s'%args.img_size, '%s'%args.lb_ratio,  'CAP_%s_%s_%s_%s_%s'%(args.loss_lb, args.loss_ub, args.warmup_epochs, args.lb_bs, args.epochs))

    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):
    # build model
    if args.net in ['resnet50']:
        model = create_model(args.net, n_classes=args.n_classes)
    elif args.net=='mlder':
        model = resnet50_ml_decoder(num_classes=args.n_classes)

    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()

    if args.resume:
        if os.path.exists(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume))

            args.start_epoch = checkpoint['epoch']+1
            if 'state_dict' in checkpoint and 'state_dict_ema' in checkpoint:
                if args.dataset_name in ['nus']:
                    state_dict = clean_state_dict(checkpoint['state_dict_ema'])
                else:
                    state_dict = clean_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("No model or state_dicr Found!!!")

            model.load_state_dict(state_dict, strict=False)
            print(np.array(checkpoint['regular_mAP']))
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            
            del checkpoint
            del state_dict
            # del state_dict_ema
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ema_m = ModelEma(model, args.ema_decay)

    # Data loading code
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args)
    print("len(lb_train_dataset):", len(lb_train_dataset)) 
    print("len(ub_train_dataset):", len(ub_train_dataset))
    print("len(val_dataset):", len(val_dataset))

    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=args.warmup_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # optimizer
    optimizer = set_optimizer(model, args)
    args.steps_per_epoch = len(lb_train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch, epochs=args.warmup_epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    best_mAP = 0


    # Used loss
    if args.loss_lb == 'bce':
        criterion_lb = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_lb == 'asl':
        criterion_lb = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')

    
    if args.loss_ub == 'bce':
        criterion_ub = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_ub == 'asl':
        criterion_ub = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')


    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):

        torch.cuda.empty_cache()

        # train for one epoch
        if epoch < args.warmup_epochs:
            loss = train(lb_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion_lb)
        else:
            adjust_per(epoch-args.warmup_epochs, args)
            print('Pos per: %.3f Neg per: %.3f'%(args.pos_per, args.neg_per))
            pb_train_dataset = pseudo_label(ub_train_dataset, model, ema_m.module, args, logger)
            pb_train_loader = torch.utils.data.DataLoader(
                    pb_train_dataset, batch_size=args.ub_bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
            if epoch == args.warmup_epochs:
                lb_train_loader = torch.utils.data.DataLoader(
                    lb_train_dataset, batch_size=args.lb_bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                optimizer = set_optimizer(model, args)
                args.steps_per_epoch = len(pb_train_loader)
                scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, pct_start=0.2)

            loss = semi_train(lb_train_loader, pb_train_loader, model, ema_m, optimizer, scheduler, epoch, args,logger, criterion_lb, criterion_ub)

            if summary_writer:
            # tensorboard logger
                summary_writer.add_scalar('pos_per', args.pos_per, epoch)
                summary_writer.add_scalar('neg_per', args.neg_per, epoch)


        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        # evaluate on validation set
        mAP = validate(val_loader, model, args, logger)
        mAP_ema = validate(val_loader, ema_m.module, args, logger)


        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

        regular_mAP_list.append(mAP)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)


        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = max(best_regular_mAP, mAP)
            best_regular_epoch = epoch
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = max(mAP_ema, best_ema_mAP)
            best_ema_epoch = epoch

        if mAP_ema > mAP:
            mAP = mAP_ema

        is_best = mAP > best_mAP
        if is_best:
            best_epoch = epoch
            best_mAP = mAP
            state_dict = model.state_dict()
            state_dict_ema = ema_m.module.state_dict()
            save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict,
                'state_dict_ema': state_dict_ema,
                'regular_mAP': regular_mAP_list,
                'ema_mAP': ema_mAP_list,
                'best_regular_mAP': best_regular_mAP,
                'best_ema_mAP': best_ema_mAP,
                'optimizer' : optimizer.state_dict(),
            }, is_best=True, filename=os.path.join(args.output, 'best_model.pth.tar'))

        logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))


        # early stop

        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 5:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

def adjust_per(epoch, args):
    
    if epoch==0:
        args.pos_per=args.init_pos_per
        args.neg_per=args.init_neg_per
    
    args.pos_per = np.clip(args.pos_per, 0.0, 1.0)
    args.neg_per = np.clip(args.neg_per, 0.0, 0.998)
    

def set_optimizer(model, args):

    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, 'AdamW')(
            param_dicts,
            args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )

    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def pseudo_label(ub_train_dataset, model, ema_model, args, logger):

    loader = torch.utils.data.DataLoader(
        ub_train_dataset, batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    model.eval()
    outputs = []
    labels = []
    for i, ((inputs_w, inputs_s), targets) in enumerate(loader):

        inputs_w = inputs_w.cuda(non_blocking=True)
        
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
                logits_w = ema_model(inputs_w)

        outputs.append(torch.sigmoid(logits_w).detach().cpu().numpy())
        labels.append(targets.numpy())

    labels = np.concatenate(labels)
    outputs = np.concatenate(outputs)
    sorted_outputs = -np.sort(-outputs, axis=0)

    n_ub = len(outputs)

    indices = [int(x)-1 for x in args.pos_label_freq*n_ub]
    thre_vec = sorted_outputs[indices, range(outputs.shape[1])]

    pseudo_labels = (outputs>=thre_vec).astype(np.float32)


    mask = np.zeros_like(pseudo_labels)

    pos_indices = [int(x)-1 for x in args.pos_per*args.pos_label_freq*n_ub]
    pos_thre = sorted_outputs[pos_indices, range(outputs.shape[1])]

    mask[outputs>=pos_thre]=1

    
    neg_indices = [int(x)-1 for x in args.neg_per*args.neg_label_freq*n_ub]
    sorted_outputs_neg = np.sort(outputs, axis=0)
    neg_thre = sorted_outputs_neg[neg_indices, range(outputs.shape[1])]

    mask[outputs<=neg_thre]=1
    
    ub_train_imgs = ub_train_dataset.X
    pb_train_dataset = MASK_HANDLER_DICT[args.dataset_name](ub_train_imgs, pseudo_labels, mask, args.dataset_dir, transform=TransformUnlabeled_WS(args))

    ##########################################  Save (Train: OPs CPs ORs CRs)  ##########################################
    if args.save_PR:
        n_correct_pos = (labels*pseudo_labels).sum(0)
        n_pred_pos = ((pseudo_labels==1)).sum(0)
        n_true_pos = labels.sum(0)
        OP = n_correct_pos.sum()/n_pred_pos.sum()
        CP = np.nanmean(n_correct_pos/n_pred_pos)
        OR = n_correct_pos.sum()/n_true_pos.sum()
        CR = np.nanmean(n_correct_pos/n_true_pos)

        auc = np.zeros(labels.shape[1])
        for i in range(labels.shape[1]):
            auc[i] = metrics.roc_auc_score(labels[:,i], pseudo_labels[:,i])
        AUC = np.nanmean(auc)

        logger.info('Train: ')
        logger.info(' AUC: %.3f'%AUC)
        logger.info(' OP: %.3f'%OP)
        logger.info(' CP: %.3f'%CP)
        logger.info(' OR: %.3f'%OR)
        logger.info(' CR: %.3f'%CR)

        PR.append([OP, CP, OR, CR, AUC])
        np.save(os.path.join(args.output, "PR"), np.array(PR))
    #####################################################################################################################

    return pb_train_dataset

PR = []

def semi_train(lb_train_loader, ub_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion_lb, criterion_ub):

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    loss_lb = AverageMeter('L_lb', ':5.3f')
    loss_ub = AverageMeter('L_ub', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [loss_lb, loss_ub, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()
    lb_train_iter = iter(lb_train_loader)
    for i, ((inputs_w_ub, inputs_s_ub), labels_ub, mask) in enumerate(ub_train_loader):

        try:
            (_, inputs_s_lb), labels_lb = next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (_, inputs_s_lb), labels_lb = next(lb_train_iter)

        n_lb = labels_lb.shape[0]
        n_ub = labels_ub.shape[0]
        inputs = torch.cat([inputs_s_lb, inputs_s_ub], dim=0).cuda(non_blocking=True)
        labels = torch.cat([labels_lb, labels_ub], dim=0).cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)

        logits_s_lb, logits_s_ub = logits[:n_lb], logits[n_lb:]
        labels_lb, labels_ub = labels[:n_lb], labels[n_lb:]

        L_lb = criterion_lb(logits_s_lb, labels_lb).sum()

        L_ub = (criterion_ub(logits_s_ub, labels_ub)*mask).sum()


        loss = L_lb+L_ub

        # *********************************************************************************************************************

        # record loss
        loss_lb.update(L_lb.item(), inputs_s_lb.size(0))
        loss_ub.update(L_ub.item(), inputs_s_ub.size(0))
        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg

def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    loss_base = AverageMeter('L_base', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [loss_base, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, ((inputs_w, inputs_s), targets) in enumerate(train_loader):

        # ****************compute loss************************

        batch_size = inputs_w.size(0)

        inputs = torch.cat([inputs_w, inputs_s], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)

        logits_w, logits_s = torch.split(logits[:], batch_size)
        
        L_base = criterion(logits_s, targets).sum()
        
        loss = L_base

        # record loss
        loss_base.update(L_base.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)


        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outputs_sm_list = []
    targets_list = []
        
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_sm = torch.sigmoid(model(inputs))
        
        # add list
        outputs_sm_list.append(outputs_sm.detach().cpu())
        targets_list.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    labels = np.concatenate(targets_list)
    outputs = np.concatenate(outputs_sm_list)

    # calculate mAP
    mAP = function_mAP(labels, outputs)
    
    print("Calculating mAP:")  
    logger.info("  mAP: {}".format(mAP))

    return mAP


if __name__ == '__main__':
    main()

import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--network", default="network.resnet38_EDAM_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--train_list", default="voc12/EDAM_train.txt", type=str)
    parser.add_argument("--session_name", default="EDAM", type=str)
    parser.add_argument("--crop_size", default=368, type=int)
    parser.add_argument("--voc12_root", required=True, type=str)
    args = parser.parse_args()
    writer = SummaryWriter(flush_secs=10)
    model = getattr(importlib.import_module(args.network), 'Net')()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    train_dataset = voc12.data.VOC12EDAMClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(256, 512),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))

    train_data_loader = DataLoaderX(train_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': args.lr * 2, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': args.lr * 10, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': args.lr * 20, 'weight_decay': 0},
        {'params': param_groups[4], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[5], 'lr': args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step, warm_up_step=2000)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    loss_function = torch.nn.BCEWithLogitsLoss().cuda()
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.session_name):
        os.makedirs(args.session_name)

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img = pack[1].cuda(non_blocking=True)
            label = pack[2]
            label = label.view(-1).cuda(non_blocking=True)
            label_idx = pack[3]
            x = model(img, label_idx)
            loss = loss_function(x, label)

            avg_meter.add({'loss': loss.item()})
            writer.add_scalar('Train/Loss', loss.item(), optimizer.global_step)
            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], optimizer.global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)
            if optimizer.global_step % (max_step//25) == 0:
                ep = int(optimizer.global_step // 1322)
                torch.save(model.module.state_dict(), os.path.join(args.session_name, str(ep) + '.pth'))

    writer.close()


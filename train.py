import os
import config
import time
import logging
import warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from models import GIG
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, cal_loss
from datasets import get_trainval_datasets
import math

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
# cross_entropy_loss = cal_loss  # 标签平滑
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
recn_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0


# 固定随机种子
def seed_torch(seed=1231):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    seed_torch(1231)
    # Initialize saving directory
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Logging setting
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    # load dataset
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)
    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True, drop_last=True), \
                                    DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                                               num_workers=config.workers, pin_memory=True, drop_last=True)
    num_classes = train_dataset.num_classes

    # Initialize model
    logs = {}
    start_epoch = 0
    net = GIG(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)

    if config.ckpt and os.path.isfile(config.ckpt):
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])  # start from the beginning

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))
        print('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    # Use cuda
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # Optimizer, LR Scheduler
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # ModelCheckpoint
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    # TRAINING
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']
        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(epoch=epoch,
              logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar)

        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar,
                 epoch=epoch)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()
        callback.on_epoch_end(logs, net, feature_center=feature_center)
        pbar.close()


def train(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']
    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    recn_metric.reset()
    crop_metric.reset()
    drop_metric.reset()
    # begin training
    start_time = time.time()
    net.train()
    batch_info = ''
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        # obtain data for training
        X = X.to(device)
        y = y.to(device)
        # raw image
        y_pred_raw, y_pred_aux, y_pred_recn, feature_matrix, attention_map = net(X, device, True)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        # Attention Cropping and Dropping
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        # crop images forward
        y_pred_aug, y_pred_aux_aug, y_pred_aux_recn, _, _ = net(aug_images, device, False)

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        # y_pred_recn = torch.cat([y_pred_recn, y_pred_aux_recn], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                     cross_entropy_loss(y_pred_recn, y) / 3. + \
                     cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_recn_acc = recn_metric(y_pred_recn, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Recn Acc ({:.2f}, {:.2f}), Aug Acc ({:.2f}, {:.2f}), ' \
                     'Aux Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
                                                       epoch_recn_acc[0], epoch_recn_acc[1], epoch_crop_acc[0],
                                                       epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])
        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_recn_{}'.format(recn_metric.name)] = epoch_recn_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    global best_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    recn_metric.reset()
    drop_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)
            # Raw Image
            y_pred_raw, y_pred_aux, y_pred_recn, _, attention_map = net(X, device, True)

            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, y_pred_aux_recn3, _, _ = net(crop_images3, device, False)

            # Final prediction
            # y_pred = (y_pred_raw + y_pred_crop3) / 2.
            # y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.
            y_pred = (y_pred_raw + y_pred_crop3 + y_pred_recn) / 3.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.
            # y_pred_recn = (y_pred_recn + y_pred_aux_recn3) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)
            # recn_acc = recn_metric(y_pred_recn, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    # batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    # pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    if epoch_acc[0] > best_acc:
        best_acc = epoch_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if aux_acc[0] > best_acc:
        best_acc = aux_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    # if recn_acc[0] > best_acc:
    #     best_acc = recn[0]
    #     save_model(net, logs, 'model_bestacc.pth')
    #
    # if epoch % 10 == 0:
    #     save_model(net, logs, 'model_epoch%d.pth' % epoch)

    # batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Val Recn Acc ({:.2f}, ' \
    #              '{:.2f}), Best {:.2f}'.format(epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1],
    #                                            recn_acc[0], recn_acc[1], best_acc)
    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), ' \
                 'Best {:.2f}'.format(epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


def save_model(net, logs, ckpt_name):
    torch.save({'logs': logs, 'state_dict': net.state_dict()}, config.save_dir + 'model_bestacc.pth')


if __name__ == '__main__':
    main()

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch_geometric.data import Data
import cv2 as cv
from torch_scatter import scatter
import copy

##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


##################################
# Metric
##################################
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
                (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError(
            'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


##############################################
# Constructing graphs
##############################################
# Constructing graphs by shifting
def get_grid_adj(grid):
    edge_index = list()
    # 上偏移
    a = torch.full_like(grid, -1)
    a[:-1] = grid[1:]  # 向上偏移，第一行没了
    adj = torch.stack([grid, a], dim=-1)  # 按最后一维即列进行堆叠
    mask = adj != -1  # 返回值是True或者False
    mask = torch.logical_and(mask[..., 0], mask[..., 1])  # 逻辑与 [...,0]表示与[:,:,0]等价
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 下偏移
    a = torch.full_like(grid, -1)
    a[1:] = grid[:-1]
    adj = torch.stack([grid, a], dim=-1)
    mask = adj != -1
    mask = torch.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 左偏移
    a = torch.full_like(grid, -1)
    a[:, :-1] = grid[:, 1:]
    adj = torch.stack([grid, a], dim=-1)
    mask = adj != -1
    mask = torch.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    # 右偏移
    a = torch.full_like(grid, -1)
    a[:, 1:] = grid[:, :-1]
    adj = torch.stack([grid, a], dim=-1)
    mask = adj != -1
    mask = torch.logical_and(mask[..., 0], mask[..., 1])
    tmp = adj[mask]
    tmp = tmp.tolist()
    edge_index += tmp
    return edge_index  # edge_index是list，里面是一对一对的坐标


# Getting graph list
def get_graph_list(data, seg):
    graph_node_feature = []
    graph_edge_index = []
    for i, j in zip(range(1, 33), seg):
        # 获取节点特征
        graph_node_feature.append(data[j == i])  # list数组  list厘面包含ndarray
        # 获取邻接信息
        j = j.cpu().numpy()
        x, y = np.nonzero(j == i)  # 返回数组a中非零元素的索引值数组,第一个从行维度来描述索引值；第二个从列维度来描述索引值
        n = len(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid = torch.full((x_max - x_min + 1, y_max - y_min + 1), -1)  # 返回一个指定形状、类型和数值的数组
        x_hat, y_hat = x - x_min, y - y_min
        grid[x_hat, y_hat] = torch.arange(n)
        graph_edge_index.append(get_grid_adj(grid))  # list数组  list厘面包含list
    graph_list = []
    # 数据变换
    for node, edge_index in zip(graph_node_feature, graph_edge_index):  # 并行遍历
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        graph_list.append(Data(node, edge_index=edge_index))
    return graph_list


# Calculate cross entropy loss, apply label smoothing if needed.
def cal_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss

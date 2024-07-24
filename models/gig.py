import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d
import random

from sklearn.preprocessing import scale
from utils import get_graph_list
from torch_geometric.data import Batch
from torch_geometric import nn as gnn
import config

__all__ = ['GIG']
EPSILON = 1e-6


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, device, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()
        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))
        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', attentions, features) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)
        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)
        # l2 normalization along dimension M and C
        # feature_matrix = F.normalize(feature_matrix_raw, dim=-1)
        feature_matrix = F.normalize(feature_matrix, dim=-1).view(B, M, C)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)  # 从0-2的随即均分分布里面取值并重新赋值
        else:
            fake_att = torch.ones_like(attentions)
        # 反事实特征
        counterfactual_feature = (torch.einsum('imjk,injk->imn', fake_att, features) / float(H * W)).view(B, -1)
        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + EPSILON)
        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)

        # bap特征、反事实特征
        return feature_matrix, counterfactual_feature


# Internal graph convolution feature module
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size):
        super().__init__()
        self.gcn = gnn.SGConv(c_in, hidden_size, K=3)

    def forward(self, graph):
        h = F.relu(self.gcn(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        return h_avg


# 外部图
class GraphNetFeature(nn.Module):
    def __init__(self, emb_dims=1024, output_channels=40, k=20, num_features=768):  # output_channels对应分类数
        super(GraphNetFeature, self).__init__()
        self.emb_dims = emb_dims
        self.output_channels = output_channels
        self.k = k
        self.num_features = num_features
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)  # Dimension of embeddings
        self.conv1 = nn.Sequential(nn.Conv2d(self.num_features * 2, 512, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(512 * 2, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(1152, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(512, self.output_channels)  # 原本是256

    def forward(self, x, device):
        batch_size = x.size(0)
        x = get_graph_feature(x, device, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, device, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, device, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, device, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        x = self.linear3(x)
        return x


class GIG(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(GIG, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net
        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)
        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        # Internal graph convolution
        self.intNet = SubGcnFeature(768, config.internalFeature)  # 第一是输入维度，第二个是输出特征纬度
        self.externalNet = GraphNetFeature(768, self.num_classes, 25, self.num_features)
        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)
        logging.info(
            'GIG: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes,
                                                                                               self.M))

    # 可视化的时候用
    def visualize(self, x):
        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)
        p = self.fc(feature_matrix * 100.)
        return p, attention_maps

    def forward(self, x, device, original):
        batch_size = x.size(0)
        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]

        feature_matrix, feature_matrix_hat = self.bap(feature_maps, attention_maps)
        feature_matrix_ori = feature_matrix.view(batch_size, -1)
        # GIG Classification
        pOri = self.fc(feature_matrix_ori * 100.)
        pCAL = pOri - self.fc(feature_matrix_hat * 100.)

        # Generate Attention Map
        attention_map = []
        if self.training:
            # Randomly choose one of attention maps Ak
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        pRecn = torch.zeros_like(pOri)

        if original:
            # 求内部图特征部分
            # newX = F.max_pool2d(x, kernel_size=10, stride=10)  # 将原始图像变成44*44
            # attention_maps_new = F.upsample_bilinear(attention_maps, size=(newX.size(2), newX.size(3)))

            intFeatures = []
            for i in range(batch_size):
                seg = []
                # data = newX[i].permute(1, 2, 0)
                data = feature_maps[i].permute(1, 2, 0)
                # h, w, c = data.size()
                # data = data.view(h * w, c)
                # data_normalization = scale(data).view(h, w, c)
                # attention_maps_new_i = attention_maps_new[i]
                attention_maps_new_i = attention_maps[i]
                for j in range(self.M):
                    attention_data = attention_maps_new_i[j]  # 26*26
                    #  控制节点个数
                    num = torch.sum(attention_data > 0)
                    # median = torch.median(attention_data)
                    # mean = torch.mean(attention_data)
                    # segData = torch.zeros_like(attention_data)
                    # if num < 1000:
                    #     segData = torch.where(attention_data > 0, j + 1, 0)
                    # elif 1000 <= num:
                    #     segData = torch.where(median <= attention_data, j + 1, 0)
                    if num > 0:
                        segData = torch.where(attention_data > 0.0, j + 1, 0)
                    else:
                        segData = torch.where(attention_data == 0.0, j + 1, 0)
                    seg.append(segData)
                graph_list = get_graph_list(data, seg)  # 构造内部图
                subGraph = Batch.from_data_list(graph_list)
            ## externalFeatures = torch.cat((feature_matrix, intFeatures), -1)
            ## pRecn = self.externalNet(externalFeatures.permute(0, 2, 1), device)
                intFeature = self.intNet(subGraph.to(device))  # 内部图卷积
                intFeatures.append(intFeature)
            intFeatures = torch.stack(intFeatures)  # 内部图卷积总特征

            intFeatures = intFeatures.view(batch_size, -1)
            intFeatures = torch.sign(intFeatures) * torch.sqrt(torch.abs(intFeatures) + EPSILON)
            intFeatures = F.normalize(intFeatures, dim=-1).view(batch_size, self.M, -1)

            pRecn = self.externalNet(intFeatures.permute(0, 2, 1), device)

        # bap分类结果、反事实分类结果、gig结果、bap特在、随机注意力
        return pOri, pCAL, pRecn, feature_matrix_ori, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            print('%s: All params loaded' % type(self).__name__)
        else:
            print('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(GIG, self).load_state_dict(model_dict)

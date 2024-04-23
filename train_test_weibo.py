import time
from transformers import BertConfig, BertModel, BertTokenizer
from PIL import Image
from thop import profile
from torch_geometric.data import Data
import re
import sys
from time import *
import gc
import random
import graph_part.config_file as config_file
import threading
import json
import pickle
import argparse
import math
import torch.nn.init as init
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.utils as utils
import abc
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 101和152代码里有版本对应问题，这个resnet-0.1包查不到版本依赖所以算了
# 有时间研究可以找一下版本对应

parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()


def process_config(config):
    for k, v in config.items():
        config[k] = v[0]
    return config


config = process_config(config_file.config)  # 读取基础的config文件


class PGD(object):  # 对抗扰动
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):  # model是下面定义的class NeuralNetwork
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad:

                if self.emb_name in name:

                    if is_first_attack:
                        self.emb_backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        r_at = self.alpha * param.grad / norm  # alpha=1.8
                        param.data.add_(r_at)  # 加上随机扰动
                        param.data = self.project(
                            name, param.data, self.epsilon)

    def restore(self):  # 恢复参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:  # 将加上扰动后的结果限定在一定范围内，epsilon=6
            r = epsilon * r / torch.norm(r)  # 如果向量的l2 norm大于阈值则让他等于阈值
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):  # 恢复梯度
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class resnet152():
    def __init__(self):
        self.model = models.resnet152(pretrained=True).cuda()
        self.model.fc = nn.Linear(2048, 300).cuda()
        # 修改最后一层全连接层的维度，变为我们需要的维度
        torch.nn.init.eye_(self.model.fc.weight)
        # 对于全连接层引入合适的先验信息
        self.path = os.getcwd() + '/dataset/weibo/weibo_images/weibo_images_all/'
        self.trans = self.img_trans()

    def img_trans(self):
        # transforms.Compose主要的作用是串联多个图形变换的操作
        transform = transforms.Compose([
            transforms.Resize(256),  # 改变大小
            transforms.CenterCrop(224),  # 中心裁剪
            transforms.ToTensor(),  # 将图像转换为张量信息
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])  # imageset训练得到的均值以及标准差数据对通道进行相应的操作
        return transform

    def forward(self, x_id):
        img_list = []
        for imgid in x_id.cpu().numpy():
            imgpath = self.path + str(imgid) + '.jpg'
            im = np.array(self.trans(Image.open(imgpath)))
            im = torch.from_numpy(np.expand_dims(im, axis=0)).to(torch.float32)
            img_list.append(im)
        batch_img = torch.cat(img_list, dim=0).cuda()
        # 将一个批次的图像转化为可以输入模型的格式并且返回模型输出，其中数据，全连接层，resnet模型都被放置在gpu上计算
        with torch.no_grad():
            img_output = self.model(batch_img)
        return img_output


class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.is_layer_norm = is_layer_norm
        if self.is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)
        V_att = Q_K_score.bmm(V)
        return V_att
    # 在特征增强公式的输入中，要么三个输入张量尺寸相同，要么是后两个相同，旨在对Q代表特征张量增强
    # 即k_len == v_len，所以不用在意三维张量中间维度的大小

    # QKV矩阵是特征矩阵，不同的增强方式这几个输入矩阵的内容不同，基本的形状控制要在不同模块之间保持一致。
    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        # [batchsize,len,inputsize] * [inputsize,n_heads*dim_k] 在进行输入的时候只需要将inputesize进行对齐即可
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, k_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(
            bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(
            bsz*self.n_heads, k_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(
            bsz*self.n_heads, k_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(
            bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))  # 通过一个尺寸调整矩阵完成最后的修改
        # [bsz,q_len,inputsize]
        return output

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)  # 创建临时变量接受多头注意力模型的返回信息

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X  # 残差
        return output


class EncoderBlock(nn.Module):  # 处理数据结构改变带来的问题，为四个文本维度共享,配合bert的提取信息完成基本的文本特征强化
    # 所有文本相关嵌入的权重共享的多层感知机，对bert输出中存在各向异性的输出进行处理和形变相关问题的处理
    def __init__(self, input_dim=768, output_dim=300, hidden_dim_1=300, hidden_dim_2=450, attn_drop=0.16):
        super(EncoderBlock, self).__init__()
        self.attn_drop = attn_drop  # 待调整
        embedding_weights = config['text_weight']
        embedding_weights_1 = config['text_weight_1']
        embedding_weights_2 = config['text_weight_2']
        embedding_weights_3 = config['text_weight_3']
        embedding_weights_4 = config['node_embedding']
        self.newid2index = config['newid2index']
        self.embedding_layer = nn.Embedding(
            num_embeddings=1467, embedding_dim=input_dim, padding_idx=0, _weight=embedding_weights)
        self.embedding_layer_1 = nn.Embedding(
            num_embeddings=1467, embedding_dim=input_dim, padding_idx=0, _weight=embedding_weights_1)
        self.embedding_layer_2 = nn.Embedding(
            num_embeddings=1467, embedding_dim=input_dim, padding_idx=0, _weight=embedding_weights_2)
        self.embedding_layer_3 = nn.Embedding(
            num_embeddings=1467, embedding_dim=input_dim, padding_idx=0, _weight=embedding_weights_3)
        # print(torch.from_numpy(embedding_weights_4).shape)
        self.graph_embedding = nn.Embedding(
            num_embeddings=6963, embedding_dim=input_dim, padding_idx=0, _weight=torch.from_numpy(embedding_weights_4[0]))
        self.graph_embedding_1 = nn.Embedding(
            num_embeddings=6963, embedding_dim=input_dim, padding_idx=0, _weight=torch.from_numpy(embedding_weights_4[1]))
        self.graph_embedding_2 = nn.Embedding(
            num_embeddings=6963, embedding_dim=input_dim, padding_idx=0, _weight=torch.from_numpy(embedding_weights_4[2]))
        self.graph_embedding_3 = nn.Embedding(
            num_embeddings=6963, embedding_dim=input_dim, padding_idx=0, _weight=torch.from_numpy(embedding_weights_4[3]))

        self.linear_1 = nn.Linear(input_dim, hidden_dim_1)
        self.linear_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.linear_3 = nn.Linear(hidden_dim_2, output_dim)
        self.dropout = nn.Dropout(attn_drop)
        self.relu = nn.ReLU()

        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.linear_1.weight)
        init.xavier_normal_(self.linear_2.weight)
        init.xavier_normal_(self.linear_3.weight)

    def forward(self, layer_id, X_id=0):
        if torch.is_tensor(X_id):  # 文本接口
            id = []
            for x_id in X_id.cpu().numpy():
                id.append(self.newid2index[x_id])
            X_id = torch.tensor(id).cuda()
            if (layer_id == 0):
                X_ = self.embedding_layer(X_id).to(torch.float32)
            elif (layer_id == 1):
                X_ = self.embedding_layer_1(X_id).to(torch.float32)
            elif (layer_id == 2):
                X_ = self.embedding_layer_2(X_id).to(torch.float32)
            elif (layer_id == 3):
                X_ = self.embedding_layer_3(X_id).to(torch.float32)
        elif X_id == 0:  # 不规范的调用接口，只处理两种情况，图接口
            if layer_id == 4:
                X_ = self.graph_embedding(torch.arange(
                    0, 6963).long().cuda()).to(torch.float32)
            elif layer_id == 5:
                X_ = self.graph_embedding(torch.arange(
                    0, 6963).long().cuda()).to(torch.float32)
            elif layer_id == 6:
                X_ = self.graph_embedding(torch.arange(
                    0, 6963).long().cuda()).to(torch.float32)
            elif layer_id == 7:
                X_ = self.graph_embedding(torch.arange(
                    0, 6963).long().cuda()).to(torch.float32)

        residual = self.relu(self.linear_1(X_))
        x_ = self.relu(self.dropout(self.linear_2(residual)))
        x_ = self.linear_3(x_)+residual

        return x_


class GraphAttentionLayer(nn.Module):  # 单层注意力工具模块
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 进入的特征维度
        self.out_features = out_features  # 输出的特征维度
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # [(2*out_feature)*1]
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(
            size=(2 * in_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        # mm矩阵乘法 [n*d] * [d*out_feature]=[n*out_feature]
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        # [n*out_feature]*[out_feature*1]=[n*1]
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        # [n*out_feature]*[out_feature*1]=[n*1]
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)  # [n,n]

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(
            negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)  # [n,n]#[n,d]
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat(
            [h_prime, h_prime_negative], dim=1)  # [n,2d]
        new_h_prime = torch.mm(h_prime_double, self.wtrans)  # [n,2d]#[2d,d]

        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Signed_GAT(nn.Module):
    def __init__(self, encoder_block, cosmatrix, nfeat, uV, original_adj, hidden=16,
                 nb_heads=4, n_output=300, dropout=0, alpha=0.3):
        # node_embedding应用config文件中读取出来的图信息，cosmatrix在调用函数中计算
        # nfeat是输入特征向量的维度，统一规定为300
        # uV通过node_embedding定义
        # original_adj是在数据提取阶段从稀疏矩阵转换过来的原始邻接矩阵
        # 作用：基本数据处理，余弦相似度强化邻接矩阵，然后将这些参数传递给多个注意力层
        super(Signed_GAT, self).__init__()
        self.dropout = dropout
        self.uV = uV

        self.user_tweet_embedding = encoder_block
        self.linear = nn.Linear(1200, 300)
        self.relu = nn.ReLU()
        self.original_adj = torch.from_numpy(
            original_adj.astype(np.float64)).cuda()
        self.potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(
            cosmatrix), torch.zeros_like(cosmatrix)).cuda()
        self.adj = self.original_adj + self.potentinal_adj
        self.adj = torch.where(self.adj > 0, torch.ones_like(
            self.adj), torch.zeros_like(self.adj))
        self.attentions = [GraphAttentionLayer(nfeat, n_output, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nb_heads)]
        # 多头注意力机制
        # 分别创建多个注意力层，各自学习
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(
                i), attention)  # 作为子层学习到关联模型的参数
        self.out_att = GraphAttentionLayer(
            nfeat * nb_heads, n_output, dropout=dropout, alpha=alpha, concat=False)

        self.__init_weights__()

    def __init_weights__(self):

        init.xavier_normal_(self.linear.weight)

    def forward(self, X_tid, epoch):
        X_0 = self.user_tweet_embedding(layer_id=4)
        X_1 = self.user_tweet_embedding(layer_id=5)
        X_2 = self.user_tweet_embedding(layer_id=6)
        X_3 = self.user_tweet_embedding(layer_id=7)
        X = self.relu(self.linear(torch.cat((X_0, X_1, X_2, X_3), dim=1)))
        if epoch == config['epochs']:
            save_path = os.getcwd()+'/dataset/weibo/weibo_files/node_embedding_pretrained.pt'
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(X, save_path)
        # 二维的节点特征向量矩阵[,300],学习整个社交图的特征
        x = F.dropout(X, self.dropout, training=self.training)
        adj = self.adj.to(torch.float32)
        x = torch.cat([att(x, adj)
                      for att in self.attentions], dim=1)  # [n,4d]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.out_att(x, adj))  # [n,d],n等同于全部节点数量

        return x[X_tid]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    @abc.abstractmethod
    def forward(self):
        pass

    def IdTrans(self, X_id):  # 将id转换为图像id
        newid2num = config['transid']
        im = []
        for id in X_id.cpu().numpy():
            im.append(int(newid2num[id]))
        return torch.tensor(im)

    def fit(self, x_train_id, y_train, x_dev_id, y_dev):  # 整理多轮次的训练测试
        if torch.cuda.is_available():
            self.cuda()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=8e-5, weight_decay=0)  # 设定模型参数的学习率
        x_train_id = torch.LongTensor(x_train_id)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(x_train_id, y_train)
        dataloader = DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=False)
        loss = nn.CrossEntropyLoss()
        pgd_block = PGD(self, emb_name='embedding_layer', epsilon=6, alpha=1.8)
        # 这里是对文本嵌入层参数做对抗训练，这里可以改成对enoferblock进行对抗干扰
        # 打算用那个图像增强文本信息处理的话，pgd对抗这里的细节还需要仔细想一下
        for epoch in range(config['epochs']):
            print("\nEpoch ", epoch + 1, "/", config['epochs'])
            self.train()
            for i, data in enumerate(dataloader):  # 转化成批次大小对象的迭代器了，迭代对象是dataset组
                total = len(dataloader)
                batch_x_id, batch_y = (item.cuda(device=self.device)
                                       for item in data)  # 解绑
                self.batch_dealer(batch_x_id, batch_y, loss,
                                  i, epoch+1, total, pgd_block)
                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(
                        self.parameters(), max_norm=self.init_clip_max_norm)
            self.batch_evaluate(x_dev_id, y_dev)

    def batch_dealer(self, x_id, y, loss, i, epoch, total, pgd_block):
        self.optimizer.zero_grad()
        logit_original, dist_org = self.forward(x_id, epoch=epoch)
        loss_classify = loss(logit_original, y)
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist_org[0], dist_org[1])
        # 回答问题1：不同模训练一个注意力模块可以训练一个模块有足够丰富的表意捕捉，当然分别训练多个也可以
        # 回答问题2：损失输出不是用来使用的而是用来衡量损失，并且将梯度反向传播给之前所有参数的
        loss_defence = 1.8*loss_classify+1.0*loss_dis  # 设置两个超参数衡量判定损失和对齐损失
        loss_defence.backward()

        K = 1  # 三轮针对文本嵌入层的参数对抗
        # 解析：
        # 多次循环中不断添加扰动之后根据新的损失修改参数，最后一轮根据之前存储的梯度和对抗的参数进行一次前向反向
        # 这次附带K次对抗对参数产生的影响信息的梯度被保留，并且参与参数更新
        pgd_block.backup_grad()
        for t in range(K):
            pgd_block.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_block.restore_grad()

            loss_adv, dist = self.forward(x_id)
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_block.restore()
        self.optimizer.step()
        corrects = (torch.max(logit_original, 1)[
                    1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total, loss_defence.item(), accuracy, corrects, y.size(0)))

    def batch_evaluate(self, x_dev_id, y_dev):
        x_dev_id = torch.LongTensor(x_dev_id)
        y_dev = torch.LongTensor(y_dev)
        y_pred = self.predicter(x_dev_id)
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
        torch.save(self.state_dict(), config['save_path'])
        # 根据参数计算对应的损失，对模型结果进行评估
        print(classification_report(y_dev, y_pred,
              target_names=config['target_names'], digits=5))
        print("Val set acc:", acc)
        print("Best val set acc:", self.best_acc)
        print("saved model at ", config['save_path'])

    def predicter(self, x_test_id):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        x_test_id = torch.LongTensor(x_test_id)
        dataset = TensorDataset(x_test_id)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_id = data[0].cuda(device=self.device)
                logits, dist = self.forward(batch_x_id)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred


class EXP(NeuralNetwork):
    def __init__(self, adj, original_adj):
        super(EXP, self).__init__()
        self.uV = adj.shape[0]
        dropout_rate = config['dropout']
        self.encoder_block = EncoderBlock()
        # 文本嵌入
        self.tv_attention = TransformerBlock(input_size=300)
        # 专门用于进行图像对于文本的增强
        self.ii_attention = TransformerBlock(input_size=300)
        self.gg_attention = TransformerBlock(input_size=300)
        self.mh_attention = TransformerBlock(input_size=300)
        # 其他所有注意力过程公用

        self.image_embedding = resnet152()
        self.cosmatrix = self.calculate_cos_matrix()
        self.gat_relation = Signed_GAT(encoder_block=self.encoder_block, cosmatrix=self.cosmatrix, nfeat=300,
                                       uV=self.uV, nb_heads=2,
                                       original_adj=original_adj, dropout=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc_tv = nn.Linear(1200, 300)
        self.fc3 = nn.Linear(1800, 900)
        self.fc4 = nn.Linear(900, 600)
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(
            in_features=300, out_features=config['num_classes'])
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.fc_tv.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.alignfc_g.weight)
        init.xavier_normal_(self.alignfc_t.weight)

    def bert_whitening(self, vecs):
        # https://kexue.fm/archives/8069
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W, -mu

    def calculate_cos_matrix(self):  # 根据所有节点的嵌入特征计算矩阵保存任意两个节点之间的余弦相似度
        load_path = os.path.dirname(
            os.getcwd()) + "/dataset/weibo/weibo_files/node_embedding_pretrained.pt"
        if os.path.exists(load_path):
            A = config['node_embedding_pretrained']
        else:
            A = torch.from_numpy(config['node_embedding'][0])

        kernel, bias = self.bert_whitening(A)
        whitened_embeddings = np.dot(A + bias, kernel)
        a, b = torch.from_numpy(whitened_embeddings), torch.from_numpy(whitened_embeddings.T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def forward(self, x_id, epoch=0):  # id是张量格式
        bsz = x_id.shape[0]

        x_id_trans = self.IdTrans(x_id).cuda()
        text_embedding_0 = self.encoder_block(layer_id=0, X_id=x_id)
        text_embedding_1 = self.encoder_block(layer_id=1, X_id=x_id)
        text_embedding_2 = self.encoder_block(layer_id=2, X_id=x_id)
        text_embedding_3 = self.encoder_block(layer_id=3, X_id=x_id)
        # encoderblock 四组文本嵌入
        iembedding = self.image_embedding.forward(x_id_trans)  # resnet152 视觉嵌入

        gc.collect()
        torch.cuda.empty_cache()
        rembedding = self.gat_relation.forward(x_id, epoch)  # Signed GAT 图嵌入

        # [64,300]
        # [64,300]
        # [64,300]

        text_embedding_0 = self.tv_attention(text_embedding_0.view(
            bsz, -1, 300), iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300))
        text_embedding_1 = self.tv_attention(text_embedding_1.view(
            bsz, -1, 300), iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300))
        text_embedding_2 = self.tv_attention(text_embedding_2.view(
            bsz, -1, 300), iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300))
        text_embedding_3 = self.tv_attention(text_embedding_3.view(
            bsz, -1, 300), iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300))
        self_att_i = self.ii_attention(iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300),
                                       iembedding.view(bsz, -1, 300))
        self_att_g = self.gg_attention(rembedding.view(bsz, -1, 300), rembedding.view(bsz, -1, 300),
                                       rembedding.view(bsz, -1, 300))
        # 共享一个注意力模块，图像特征分别增强文本特征

        text_enhanced = self.relu(self.fc_tv(torch.cat(
            (text_embedding_0, text_embedding_1, text_embedding_2, text_embedding_3), dim=2)))

        align_text = self.alignfc_t(text_enhanced).view(bsz, 300)
        align_rembedding = self.alignfc_g(self_att_g).view(bsz, 300)
        dist = [align_text, align_rembedding]
        self_att_t = text_enhanced.view(bsz, -1, 300)
        # self_att_t = align_text.view(bsz,-1,300)
        # self_att_g = align_rembedding.view(bsz,-1,300)
        co_att_tg = self.mh_attention(
            self_att_t, self_att_g, self_att_g).view(bsz, 300)
        co_att_gt = self.mh_attention(
            self_att_g, self_att_t, self_att_t).view(bsz, 300)
        co_att_ti = self.mh_attention(
            self_att_t, self_att_i, self_att_i).view(bsz, 300)
        co_att_it = self.mh_attention(
            self_att_i, self_att_t, self_att_t).view(bsz, 300)
        co_att_gi = self.mh_attention(
            self_att_g, self_att_i, self_att_i).view(bsz, 300)
        co_att_ig = self.mh_attention(
            self_att_i, self_att_g, self_att_g).view(bsz, 300)
        att_feature = torch.cat(
            (co_att_tg, co_att_gt, co_att_ti, co_att_it, co_att_gi, co_att_ig), dim=1)

        a1 = self.relu(self.dropout(self.fc3(att_feature)))

        a1 = self.relu(self.fc4(a1))

        a1 = self.relu(self.fc1(a1))

        a1 = self.dropout(a1)

        output = self.fc2(a1)

        return output, dist


def load_dataset():
    print('data_process: Start')
    pre = os.getcwd() + "/dataset/weibo/weibo_files"
    X_train_tid, _, y_train, _, adj = pickle.load(
        open(pre + "/train.pkl", 'rb'))
    X_dev_tid, _, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, _, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['node_embedding'] = pickle.load(
        open(pre + "/node_embedding.pkl", 'rb'))
    if os.path.exists(pre + "/node_embedding_pretrained.pt"):
        config['node_embedding_pretrained'] = torch.load(
            open(pre + "/node_embedding_pretrained.pt", 'rb'))
    config['text_weight'] = torch.load(open(pre + '/weight.pt', 'rb'))
    config['text_weight_1'] = torch.load(open(pre + '/weight_1.pt', 'rb'))
    config['text_weight_2'] = torch.load(open(pre + '/weight_2.pt', 'rb'))
    config['text_weight_3'] = torch.load(open(pre + '/weight_3.pt', 'rb'))
    with open(pre + '/new_id_dic.json', 'r') as f:  # 存储文本id和平台用户id的键值对
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))

    mid2num = {}
    id_list = []
    path = os.getcwd()+'/dataset/weibo/weibocontentwithimage/original-microblog/'
    for file in os.listdir(path):
        mid2num[file.split('_')[-2]] = file.split('_')[0]  # 存储对应关系
        id_list.append(file.split('_')[0])  # 存储图片id列表

    index = 0
    num2index = {}
    for i in id_list:
        num2index[i] = index
        index += 1
    config['num_toindex'] = num2index  # 存储图像id和张量下标的对应关系

    newid2num = {}  # new_id->imid
    newid2index = {}  # new_id->index(post)
    mid2index = {}  # mid(post)->index
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]  # 存储num
        # num取对应index和newid组成键值对
        mid2index[newid2mid[id]] = newid2index[id] = num2index[newid2num[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
        mid2index[newid2mid[id]] = newid2index[id] = num2index[newid2num[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
        mid2index[newid2mid[id]] = newid2index[id] = num2index[newid2num[id]]
    config['transid'] = newid2num  # resnet用
    config['newid2index'] = newid2index  # 文本嵌入用
    config['mid2index'] = mid2index  # mid->index

    print('data_load: Done')
    return X_train_tid,  y_train, \
        X_dev_tid,  y_dev, \
        X_test_tid, y_test, adj


def load_original_adj(adj):  # 将adj节点稀疏矩阵转化为original_adj邻接矩阵
    pre = os.getcwd() + '/dataset/weibo/weibo_files/'
    path = os.path.join(pre, 'original_adj')
    with open(path, 'r') as f:
        original_adj_dict = json.load(f)  # 里面存了每行，有哪些列是1
    # o_a_d输出出来是这样的{'3461':[1668], '3647':[3012,2330,1849...]..}是稀疏矩阵的形式
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():  # 根据输入的adj格式建立邻接矩阵并且遍历稀疏矩阵
        v = [int(e) for e in v]
        # 可以一次性把v这个列表里面的v_n处的 original_adj[i,v_n] 设置为 1
        original_adj[int(i), v] = 1
    return original_adj  # 初始邻接矩阵形成供给后续使用


def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")
    res_dir = 'exp_result'  # 当前目录下如果不存在这个文件夹（目录）则创建
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)  # 再创建一级，实际上是pheme文件夹
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)  # 再创建一级
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(
        res_dir, 'best_model_in_each_config')
    # 又创建了一级，并且保留为config文件的一部分
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)  # 这里根据前面args的信息进一步组装文件名称，并且存储到config文件中
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):  # 不知道为什么又检查了一遍前面的目录结构
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):  # 根据上面的那个建立文件了
        os.system('del {}'.format(config['save_path']))  # savepath就是详细的文件存储位置

    X_train_tid,  y_train, \
        X_dev_tid, y_dev, \
        X_test_tid,  y_test, adj = load_dataset()
    original_adj = load_original_adj(adj)
    nn = model(adj, original_adj)
    # 新建了一个自己MFAN类型的对象，并将已经处理好的数据传输给他，这时候config文件也已经基本完善了

    nn.fit(X_train_tid, y_train, X_dev_tid,  y_dev)

    y_pred = nn.predicter(X_test_tid)
    res = classification_report(
        y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2 = {}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    print(res)
    return res  # 程序结束


seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
model = EXP
train_and_test(model)

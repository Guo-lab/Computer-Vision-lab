from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F




#@ IN: 2500个三坐标点
#@ OUT: 获得一个3×3的变换矩阵
# 对点云的姿态进行校正，
# 而该变换矩阵就是根据点云特性，做出一个刚体变换，使点云处于一个比较容易检测的姿态
## Input Transform
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        # 先对输入经过三级卷积核为1×1的卷积处理得到1024通道的数据
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # 再经过全连接处映射到九个数据，最后调整为3×3
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x





# 对特征的一种校正，一种广义的位姿变换
## Feature Transform (Similar to Input Transform)
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x




























#@ 点云特征提取网络：global feature 和 point features
#% （1）经过STN3d获得3×3矩阵，乘以点云做完位姿变换，经过多层感知机（与卷积核边长为1的卷积操作本质一样）
#% （2）再乘以经过STNkd获得的64×64的矩阵，完成位姿变换，再经过多层感知机
#%  n * 3 =InputTransform=> n * 3 =MLP=> n * 64 =FeatureTransform=> n * 64 =MLP=> n * 1024 
#%        =MAXPOOLING=> global feature1024
###################################################
#################### MY WORK ######################
###################################################
class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        #TODO
        super(PointNetfeat, self).__init__()
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        self.stn3d = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        if self.feature_transform:
            self.stnkd = STNkd(k=64)


    def forward(self, x):
        #TODO
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        trans = self.stn3d(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))

        # 进行 Feature Transform
        if self.feature_transform:
            trans_feat = self.stnkd(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        # 否则不    
        else:
            trans_feat = None

        # 保存记录x
        pointfeat = x
        
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.bn3(self.conv3(x))
        
        # Maxpool
        x = torch.max(x, 2, keepdim=True)[0]
        # 调整 x 的形状
        x = x.view(-1, 1024)
        
    
    #*  根据PointNetCls():
    # * x, trans, trans_feat = self.feat(x)    
        if self.global_feat:
            return x, trans, trans_feat
        # Point features
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
  
###################################################






























#% 最大池化和多层感知机进行分类了，经过全连接分成k类，根据概率来判别究竟属于哪一类
class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        # 防止过拟合
        self.dropout = nn.Dropout(p=0.3)

        # 归一化防止梯度爆炸与梯度消失
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 完成网络主体部分
        x, trans, trans_feat = self.feat(x)

        # 经过三个全连接层（多层感知机）映射成k类
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        # 返回的是该点云是第ki类的概率
        return F.log_softmax(x, dim=1), trans, trans_feat








#% 分割网络是借用了分类网络的两部分，分别是64通道和1024通道，堆积在一起形成1088通道的输入
#% 经过多层感知机输出了结果m通道的结果，
#% m代表类的个数，也就是每个点属于哪一类，实际上分割是在像素级或者点级的分类
class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        x, trans, trans_feat = self.feat(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        
        return x, trans, trans_feat









def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

















if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
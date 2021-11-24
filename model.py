
import torch
import torch.nn as nn
import torch.nn.functional as F
#x输入是四个参数
#[100,1,161,101]
# 100 batch_size,一次100个音频
# 1不清楚
# 161不清楚
# 101 frames帧数

class ConvNet(nn.Module):
    # CNN
    def __init__(self):
        super(ConvNet, self).__init__()
        self.l1 = nn.Linear(101, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 64)
        self.l4 = nn.Linear(64, 6)

    def forward(self, x):
        x=self.l1(x)
        x = F.relu(x)
        x=self.l2(x)
        x = F.relu(x)
        x=self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        return x

        '''
        ###############################################################
        do the forward propagation here
        x: model input with shape:(batch_size, frame_num, feature_size)
        frame_num is how many frame one wav have
        feature_size is the dimension of the feature
        ###############################################################
        '''

class FcNet(nn.Module):
    # DNN
    def __init__(self):
        super(FcNet, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 6)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
        '''
        ###############################################################
        do the forward propagation here
        x: model input with shape:(batch_size, frame_num, feature_size)
        frame_num is how many frame one wav have
        feature_size is the dimension of the feature
        ###############################################################
        '''

        
# 建立VGG卷积神经的模型层
def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':  # maxpool 池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 卷积层
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]  # avgPool 池化层
    return nn.Sequential(*layers)


# 各个VGG模型的参数
cfg = {
    'VGG':[32,'M',64,'M',128,'M',128,'M'],
    'VGG11': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG13': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG16': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'VGG19': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}


# VGG卷积神经网络
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])  # VGG的模型层
        print(self.features)
        self.fc1 = nn.Linear(7680, 256)            # 7680,512
        self.fc2 = nn.Linear(256, 6)     # 输出类别5，由于全量比较大，这里只选择前5个类别

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # flatting
#         print(out.size())
        out = self.fc1(out)  # 线性层
        out = self.fc2(out)  # 线性层
        return F.log_softmax(out, dim=1)  # log_softmax 激活函数

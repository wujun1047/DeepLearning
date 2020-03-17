# _*_ coding : UTF-8 _*_
# 开发团队    : xxxx
# 开发人员    : wujunyang
# 开发时间    : 2020/3/16 11:56 上午
# 文件名称    : resnet.py
# 开发工具    : PyCharm

import torch
import torch.nn as nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):

    explosion = 1  # 用于控制是否需要扩大输出通道

    """
    in_planes：这个block的输入通道数（上一个block的输出）
    planes：这个block的初始通道数，不命名为out_planes是因为它可能和
            输出通道数(= in_planes * explosion)可能不一样。
    stride：用于控制是否需要降采样。如果这个block是stage的第一个block，需要缩小图片大小，
            则stride需要设置为2。
    norm_layer：bn层，设成参数便宜调整
    """
    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.stride = stride
        self.downsample = None
        # 输入通道数和输出通道数不一致，则需要修改通道数后才能相加
        if in_planes != planes * BasicBlock.explosion:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes * BasicBlock.explosion, stride),
                                            nn.BatchNorm2d(planes * BasicBlock.explosion))

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)  # relu只需一个，因为relu不需要学习参数，可以共用

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):

    explosion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.stride = stride
        self.downsample = None
        if in_planes != planes * Bottleneck.explosion:
            self.downsample = nn.Sequential(conv1x1(in_planes, planes * Bottleneck.explosion, stride),
                                            nn.BatchNorm2d(planes * Bottleneck.explosion))

        self.conv1 = conv1x1(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.conv3 = conv1x1(planes, planes * Bottleneck.explosion)
        self.bn3 = norm_layer(planes * Bottleneck.explosion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        # stage 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # stage 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, block, layers[0], 1)
        # stage 3
        self.layer2 = self._make_layer(64 * block.explosion, 128, block, layers[1])
        # stage4
        self.layer3 = self._make_layer(128 * block.explosion, 256, block, layers[2])
        # stage5
        self.layer4 = self._make_layer(256 * block.explosion, 512, block, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.explosion, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def _make_layer(self, in_planes, planes, block, count, stride=2):
        blocks = [block(in_planes, planes, stride)]  # 第一个block可能需要降采样
        for i in range(1, count):
            blocks.append(block(planes * block.explosion, planes))
        return nn.Sequential(*blocks)


resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
resnet34 = ResNet(BasicBlock, [3, 4, 6, 3])
resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])
resnet152 = ResNet(Bottleneck, [3, 8, 36, 3])

# 和官方模型对比
import torchvision.models as models
from torchsummary import summary
# inputsize = (3, 224, 224)
# summary(resnet152, inputsize)
# summary(models.resnet152(), inputsize)

# 加载官方的预训练模型（定义的参数名称要和官方版本一致）
from torchvision.models.utils import load_state_dict_from_url
def load_pretrained(model, url):
    state_dict = load_state_dict_from_url(url, progress=True)
    model.load_state_dict(state_dict)
load_pretrained(resnet18, model_urls['resnet18'])

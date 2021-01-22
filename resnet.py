# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class ResNet18_subcenter(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, k=3, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet18_subcenter network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        self._k = k
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes * self._k, bias=False)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        feat = x.view(N, self._n_classes, self._k)
        x = self.maxpool(feat)
        x = x.view(N, self._n_classes)
        return x, feat

class ResNet18_ss(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet18_ss network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes)
        self.rot_pred = nn.Linear(in_features=512, out_features=4)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        rot_pred = self.rot_pred(x)
        x = self.fc(x)
        return x, rot_pred

class ResNet50_subcenter(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, k=3, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet50_sub network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._k = k
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes * self._k, bias=False)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = self.fc(x)
        feat = x.view(N, self._n_classes, self._k)
        x = self.maxpool(feat)
        x = x.view(N, self._n_classes)
        return x, feat

if __name__ == '__main__':
    net = ResNet18_subcenter()
    x = torch.rand(64, 3, 448, 448)
    y = net(x)
    print(y.shape)

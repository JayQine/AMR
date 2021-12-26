import torch
import torch.nn as nn
import torch.nn.functional as F

from misc import torchutils
from net import resnet50


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                        self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)


        self.resnet50_2 = resnet50.resnet50(pretrained=True, use_amm=True, strides=(2, 2, 2, 1))
        self.stage2_1 = nn.Sequential(self.resnet50_2.conv1, self.resnet50_2.bn1, self.resnet50_2.relu, self.resnet50_2.maxpool,
                                        self.resnet50_2.layer1)

        self.stage2_2 = nn.Sequential(self.resnet50_2.layer2)
        self.stage2_3 = nn.Sequential(self.resnet50_2.layer3)
        self.stage2_4 = nn.Sequential(self.resnet50_2.layer4)

        self.classifier2 = nn.Conv2d(2048, 20, 1, bias=False)

        
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.backbone2 = nn.ModuleList([self.stage2_1, self.stage2_2, self.stage2_3, self.stage2_4])
        self.newly_added = nn.ModuleList([self.classifier, self.classifier2])
        

    def forward(self, x):

        x_ori = x.clone()

        # # branch1

        x = self.stage1(x).detach()
        x = self.stage2(x).detach()
        x = self.stage3(x).detach()
        x = self.stage4(x).detach()

        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x).detach()
        x = x.view(-1, 20)

        # # branch2

        x2 = self.stage2_1(x_ori).detach()
        x2 = self.stage2_2(x2)
        x2 = self.stage2_3(x2)
        x2 = self.stage2_4(x2)

        cam2 = F.conv2d(x2, self.classifier2.weight)
        cam2 = F.relu(cam2)   
        cam2 = cam2[0] + cam2[1].flip(-1)

        x2 = torchutils.gap2d(x2, keepdims=True)
        x2 = self.classifier2(x2)
        x2 = x2.view(-1, 20)

        return x, cam, x2, cam2

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        for p in self.resnet50_2.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50_2.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.backbone2.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, step=1):

        x_ori = x.clone()

        # branch1
        if step == 1:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
    
            cam1 = F.conv2d(x, self.classifier.weight)
            return cam1


        # # branch2
        if step == 2:
            x2 = self.stage2_1(x_ori)
            x2 = self.stage2_2(x2)
            x2 = self.stage2_3(x2)
            x2 = self.stage2_4(x2)

            cam2 = F.conv2d(x2, self.classifier2.weight)
            return cam2
        
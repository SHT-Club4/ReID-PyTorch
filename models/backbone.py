import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

from models.head import ClassBlock


class Resnet50_ft(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False):
        super(Resnet50_ft, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn == True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = Resnet50_ft(751, stride=1, ibn=True)
    # net.classifier = nn.Sequential()
    # print(net)
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    print(input.shape)
    output = net(input)
    print(output.shape)

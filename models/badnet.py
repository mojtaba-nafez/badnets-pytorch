import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Resent18_BadNet(torch.nn.Module):
    def __init__(self, output_num=10, pretrained=True):
        super().__init__()
        # self.norm = lambda x: (x - mu) / std
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = torch.nn.Identity()
        self.output = torch.nn.Linear(512, output_num)

    def forward(self, x):
        # x = self.norm(x)
        x = self.backbone(x)
        # z_n = F.normalize(z1, dim=-1)
        return self.output(x)

class VIT_BadNet(torch.nn.Module):
    def __init__(self, output_num=10, pretrained=True):
        super().__init__()
        # self.norm = lambda x: (x - mu) / std
        self.backbone = models.vit_b_16(pretrained=pretrained)
        self.backbone.heads = torch.nn.Identity()
        self.output = torch.nn.Linear(748, output_num)

    def forward(self, x):
        # x = self.norm(x)
        x = self.backbone(x)
        # z_n = F.normalize(z1, dim=-1)
        return self.output(x)



class Conv_BadNet(nn.Module):

    def __init__(self, input_channels, output_num):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        fc1_input_features = 800 if input_channels == 3 else 512
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc1_input_features, out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=output_num),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def BadNet(input_channels, output_num, model='resnet18'):
    if model=='resnet18':
        model = Resent18_BadNet(output_num=output_num)
    elif model=='vit':
        model = VIT_BadNet(output_num=output_num)
    elif model=='simple_conv':
        model = Conv_BadNet(input_channels=input_channels, output_num=output_num)
    return model


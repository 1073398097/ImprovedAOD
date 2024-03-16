import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary

import torch.nn.init as init

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[4,8,16,32]):
        super(PyramidPooling, self).__init__()

        self.conv=nn.Conv2d(in_channels=7, out_channels=3, kernel_size=1, stride=1)
        # 为每个池化大小创建一个池化层
        self.pool_layers = nn.ModuleList()
        for size in pool_sizes:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            # for m in self.modules( ):
            #     if isinstance(m, nn.Conv2d):
            #         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            #         torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # 对输入的特征图进行每个池化层的池化操作，并将结果拼接起来
        pool_outputs = [x]
        #
        # import pdb
        # pdb.set_trace()

        for layer in self.pool_layers:
            pooled = layer(x)
            pool_outputs.append(F.interpolate(pooled, size=x.size()[2:], mode='bilinear', align_corners=True)) #上采样

        spp_outputs=self.conv(torch.cat(pool_outputs, dim=1))

        return spp_outputs


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,Padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=Padding, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class dehaze_net(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(dehaze_net, self).__init__( )

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

        # FPC—Net
        out_channels_num=10

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_num, kernel_size=1, stride=1)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv1.bias.data.fill_(0)

        self.conv2 = nn.Conv2d(in_channels=out_channels_num, out_channels=out_channels_num, kernel_size=1, stride=1)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.conv2.bias.data.fill_(0)

        self.conv3 = nn.Conv2d(in_channels=out_channels_num*2, out_channels=out_channels_num, kernel_size=1, stride=1)
        init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.conv3.bias.data.fill_(0)

        self.conv4 = nn.Conv2d(in_channels=out_channels_num*3, out_channels=out_channels_num, kernel_size=1, stride=1)
        init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        self.conv4.bias.data.fill_(0)

        self.conv5 = nn.Conv2d(in_channels=out_channels_num*4, out_channels=out_channels, kernel_size=1, stride=1)
        init.kaiming_normal_(self.conv5.weight, mode='fan_in', nonlinearity='relu')
        self.conv5.bias.data.fill_(0)

        # 定义一个最大池化层，kernel_size 表示池化核大小，stride 表示步长，padding表示填充值
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 第二个池化层的滤波核大小为5×5，padding值为2，可以使用以下代码表示：
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        # 第三个池化层的滤波核大小为7×7，padding值为3，可以使用以下代码表示：
        self.max_pool3 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        #DS-Conv
        self.DPconv1 =DepthwiseSeparableConv2d(in_channels=in_channels,out_channels=out_channels_num, kernel_size=3, padding=1)
        self.DPconv2 =DepthwiseSeparableConv2d(in_channels=out_channels_num,out_channels=out_channels_num, kernel_size=3, padding=1)
        self.DPconv3 =DepthwiseSeparableConv2d(in_channels=out_channels_num*2,out_channels=out_channels_num, kernel_size=5, padding=1,Padding=1)
        self.DPconv4 =DepthwiseSeparableConv2d(in_channels=out_channels_num*3,out_channels=out_channels_num, kernel_size=7, padding=1,Padding=2)
        self.DPconv5 =DepthwiseSeparableConv2d(in_channels=out_channels_num*4,out_channels=out_channels, kernel_size=3, padding=1)

        #金字塔池化
        #self.pypool=PyramidPooling(in_channels=3,out_channels=out_channels)
        self.pypool = PyramidPooling(3,1)

    def forward(self, x,y):  # torch.Size([2, 3, 224, 224])

        source = []
        source.append(x)
        x_haze=y

        #经典AOD结构网络
        # x1 = self.relu(self.e_conv1(x))
        # x2 = self.relu(self.e_conv2(x1))
        #
        # concat1 = torch.cat((x1, x2), 1)
        # x3 = self.relu(self.e_conv3(concat1))
        #
        # concat2 = torch.cat((x2, x3), 1)
        # x4 = self.relu(self.e_conv4(concat2))
        #
        # concat3 = torch.cat((x1, x2, x3, x4), 1)
        # x5 = self.relu(self.e_conv5(concat3))  # x5就相当于k

        # FPC结构网络
        # x1 = self.relu(self.conv1(x))  # torch.Size([2, 8, 224, 224])
        # x2 = self.relu(self.conv2(x1))  # torch.Size([2, 8, 224, 224])
        # x2 = self.max_pool1(x2)  # torch.Size([2, 8, 224, 224])
        #
        # concat1 = torch.cat((x1, x2), 1)  # torch.Size([2, 16, 224, 224])
        # x3 = self.relu(self.conv3(concat1))
        # x3 = self.max_pool2(x3)
        #
        # concat2 = torch.cat((x1, x2, x3), 1)
        # x4 = self.relu(self.conv4(concat2))
        # x4 = self.max_pool3(x4)
        #
        # concat3 = torch.cat((x1, x2, x3, x4), 1)
        # x5 = self.relu(self.conv5(concat3))

        # 深度可分离卷积
        x1 = self.relu(self.DPconv1(x))  # torch.Size([2, 8, 224, 224])
        x2 = self.relu(self.DPconv2(x1))  # torch.Size([2, 8, 224, 224])

        concat1 = torch.cat((x1, x2), 1)  # torch.Size([2, 16, 224, 224])
        x3 = self.relu(self.DPconv3(concat1))


        concat2 = torch.cat((x1, x2, x3), 1)
        x4 = self.relu(self.DPconv4(concat2))


        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.DPconv5(concat3))

        kx = self.pypool(x5)  # torch.Size([8, 4080])



        clean_image = self.relu((kx * x_haze) - kx + 1)

        return clean_image


if __name__ == "__main__":

    net = dehaze_net( )
    # 将模型移动到GPU上（如果有可用的GPU）
    device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
    net.to(device)
    summary(net, (3, 224, 224), batch_size=1, device='cuda')

    # 定义模型
    model = dehaze_net( )
    # 总参数量
    total_params = sum(p.numel( ) for p in model.parameters( ))
    # 可训练的参数量
    trainable_params = sum(p.numel( ) for p in model.parameters( ) if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == "__main__":

    # device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
    # model = SpatialPyramidPooling(levels=[4,8,16,32]).to(device)
    # input_shape = (8, 3, 160, 120)  # 输入张量的形状
    # summary(model, input_shape=input_shape, device=device.type)

    pypool = PyramidPooling(3,1)
    tensor = torch.rand((8, 3, 160, 120))
    print(tensor.shape)
    kx=pypool(tensor)
    print(kx.shape)



import torch
import net
from torchvision import models

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)
# model = models.resnet18(pretrained=True)
# model = model.eval().to(device)
model = torch.load('E:\wangchao\AODNet_2\snapshots\Epoch45.pth')
model = model.eval().to(device)
x = torch.randn(1, 3, 256, 256).to(device)
output = model(x)
x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,                  # 要转换的模型
        x,                      # 模型的任意一组输入
        'dehazer.onnx',        # 导出的 ONNX 文件名
        opset_version=11,       # ONNX 算子集版本
        input_names=['input'],  # 输入 Tensor 的名称（自己起名字）
        output_names=['output'] # 输出 Tensor 的名称（自己起名字）
    )
import onnx

# 读取 ONNX 模型
onnx_model = onnx.load('dehazer.onnx')

# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)

print('无报错，onnx模型载入成功')

print(onnx.helper.printable_graph(onnx_model.graph))
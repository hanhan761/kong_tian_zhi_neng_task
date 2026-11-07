import torch
import torch.onnx
import network
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)

# 初始化模型并加载权重
model = network.modeling.__dict__['deeplabv3plus_resnet50'](
    num_classes=8,
    output_stride=16,
    pretrained_backbone=True
)
checkpoint = torch.load(
    'checkpoints/best_deeplabv3plus_resnet50_uavid_os16.pth',
    map_location=device  # 直接加载到目标设备
)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# 修正：确保 dummy_input 在正确设备上
dummy_input = torch.randn(1, 3, 512, 1024, device=device, dtype=torch.float32)

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "deeplabv3plus_resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)
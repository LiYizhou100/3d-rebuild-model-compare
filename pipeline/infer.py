"""统一推理接口。所有模型共用，不 import 具体模型。"""
import torch


def infer(model, lr, device="cpu"):
    """lr numpy -> sr numpy。model 需实现 infer(lr_sr, device)。"""
    model.eval()
    with torch.no_grad():
        return model.infer(lr, device=device)

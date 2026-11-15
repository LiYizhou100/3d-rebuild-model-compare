"""配准单独训练（对齐论文 Eq.3），先于 DDPM 训练。

用法：
    python reg_train.py --mode develop --fold 0 --epochs 20
"""
import argparse

import torch

import configs
from utils.Registration.Registration import RegistrationNet
from utils.Registration.train_reg import train_registration
from utils.data import load_volume
from utils.dataset.split import split_all


def main():
    ap = argparse.ArgumentParser(description="配准单独训练（γ1·L2 + γ2·平滑 + γ3·Dice）")
    ap.add_argument("--mode", default="develop", choices=["develop", "seg", "test"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)   # 论文配准 lr=1e-3
    args, overrides = ap.parse_known_args()
    configs.apply_cli(overrides)

    train_ids = split_all(args.mode)   # 该阶段全部卷（不按折，配准用全部数据训练）
    hr, lr = load_volume(train_ids[0])
    D = lr.shape[1]

    reg_net = RegistrationNet(base=configs.MODEL_CFG.get("REG_BASE", 32)).to(configs.DEVICE)
    print(f"配准训练: mode={args.mode} fold={args.fold} {len(train_ids)} 卷  epochs={args.epochs}")

    train_registration(reg_net, train_ids, args.epochs, args.lr, configs.DEVICE)

    out = configs.PARAMETER_DIR / "reg_net.pt"
    torch.save(reg_net.state_dict(), out)
    print(f"已保存配准权重 {out}")


if __name__ == "__main__":
    main()

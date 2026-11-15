"""评估：推理 + 重建指标（L1/PSNR/SSIM）+ baseline 线性插值 + 存 guess 卷。

用法：
    python evaluate.py --run exp-modelDiffusion-cfgbase-s2-foldfinal --sample-steps 20
"""
import argparse
import json

import numpy as np
import torch

import configs
from model import build_model
from infer import infer
from utils.data import load_volume
from utils.metrics import compute_all
from utils.dataset.split import split_inside


def linear_interp(lr, scale):
    """线性插值 baseline：lr (y_lr,Z,X) -> sr (y_hr,Z,X) = (y_lr-1)*scale+1。"""
    y_lr = lr.shape[0]
    outs = [lr[0]]
    for i in range(y_lr - 1):
        for k in range(1, scale):
            t = k / scale
            outs.append((1 - t) * lr[i] + t * lr[i + 1])
        outs.append(lr[i + 1])
    return np.stack(outs)


def evaluate_model(model, ids, device, sample_steps=20, guess_dir=None):
    """对 ids 卷推理 + 算 DL/baseline 重建指标。返回 (per_volume, mean_dl, mean_base)。"""
    per_volume = []
    for sid in ids:
        hr, lr = load_volume(sid)
        sr = infer(model, lr, device=device, sample_steps=sample_steps)
        base = linear_interp(lr, model.scale)
        m_dl = compute_all(sr, hr)
        m_base = compute_all(base, hr)
        per_volume.append({"id": sid, "dl": m_dl, "base": m_base})
        if guess_dir:
            np.savez_compressed(guess_dir / f"{sid}.npz",
                                sr=sr.astype(np.float32),
                                base=base.astype(np.float32),
                                hr=hr.astype(np.float32))
        print(f"{sid}: dl l1={m_dl['l1']:.6f} base l1={m_base['l1']:.6f}"
              f"  psnr={m_dl['psnr']:.2f}/{m_base['psnr']:.2f}")
    keys = ["l1", "psnr", "ssim"]
    mean_dl = {k: float(np.mean([r["dl"][k] for r in per_volume])) for k in keys}
    mean_base = {k: float(np.mean([r["base"][k] for r in per_volume])) for k in keys}
    return per_volume, mean_dl, mean_base


def main():
    ap = argparse.ArgumentParser(description="评估：加载 best.pt，推理测试集，算重建指标")
    ap.add_argument("--run", type=str, required=True, help="训练好的 run 目录名")
    ap.add_argument("--mode", default="test", choices=["develop", "seg", "test"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sample-steps", type=int, default=20, help="DDIM 采样步数（加速）")
    args, overrides = ap.parse_known_args()
    configs.apply_cli(overrides)

    run_dir = configs.PARAMETER_DIR / args.run
    cfg = json.load(open(run_dir / "config.json"))["config"]

    # 测试集（有最终测试集则用它，否则用 val）
    train_ids, val_ids, test_ids = split_inside(args.mode, args.fold)
    ids = test_ids or val_ids

    # 探测尺寸 + 重建模型 + 加载权重
    hr, lr = load_volume(ids[0])
    y_hr, y_lr = hr.shape[0], lr.shape[0]
    model = build_model(cfg["model"], y_hr, y_lr).to(configs.DEVICE)
    best = run_dir / "best.pt"
    if best.exists():
        model.load_state_dict(torch.load(best, map_location=configs.DEVICE))
        print(f"加载 {best}")
    else:
        print("⚠️ 无 best.pt，用随机权重（仅供流程测试）")

    guess_dir = run_dir / "guess"
    guess_dir.mkdir(parents=True, exist_ok=True)
    per_volume, mean_dl, mean_base = evaluate_model(
        model, ids, configs.DEVICE, sample_steps=args.sample_steps, guess_dir=guess_dir)
    json.dump({"run": args.run, "per_volume": per_volume,
               "mean_dl": mean_dl, "mean_base": mean_base},
              open(run_dir / "val" / "results.json", "w"), indent=2)
    print(f"mean_dl:   {mean_dl}")
    print(f"mean_base: {mean_base}")
    print(f"已存 {run_dir / 'val' / 'results.json'} 和 guess/")


if __name__ == "__main__":
    main()
